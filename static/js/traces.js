/**
 * Agent 执行轨迹页。
 *
 * 左栏列出 runtime/traces/ 里的每一次执行，右栏把选中的那一次画成时间线。
 *
 * 安全上有一条硬规矩：**轨迹里的每一段文本都当作不可信内容处理**。
 * thought 是大模型生成的，observation 是工具返回的（其中 search_reviews
 * 返回的是爬来的评论原文）——这些内容里出现 <script> 或 onerror= 完全是
 * 可能的。所以全程用 textContent / createElement 组装 DOM，一处 innerHTML
 * 都不用。
 */

(function () {
    'use strict';

    const runList = document.getElementById('runList');
    const detail = document.getElementById('traceDetail');
    const refreshBtn = document.getElementById('refreshBtn');

    let traces = [];
    let activeId = null;

    // ---------- 小工具 ----------
    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = String(text);
        return node;
    }

    function fmtTime(ts) {
        if (!ts) return '';
        return new Date(ts * 1000).toLocaleString('zh-CN', {
            month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit'
        });
    }

    function fmtCost(usd) {
        if (usd === null || usd === undefined) return '—';
        return usd < 0.01 ? `$${usd.toFixed(5)}` : `$${usd.toFixed(3)}`;
    }

    function statusLabel(status) {
        return {
            completed: '完成',
            budget_exceeded: '预算耗尽',
            error: '异常',
            running: '进行中'
        }[status] || status;
    }

    // ---------- 左栏 ----------
    function renderRunList() {
        runList.replaceChildren();
        if (!traces.length) {
            runList.appendChild(el('p', 'trace-hint',
                '还没有任何执行记录。跑一次 python scripts/demo_supervisor.py 就会出现。'));
            return;
        }
        traces.forEach(function (t) {
            const item = el('button', 'run-item' + (t.run_id === activeId ? ' active' : ''));
            item.type = 'button';
            item.appendChild(el('div', 'run-task', t.task || '(无任务描述)'));

            const meta = el('div', 'run-meta');
            meta.appendChild(el('span', null, fmtTime(t.started_at)));
            meta.appendChild(el('span', null, statusLabel(t.status)));
            if (t.roles && t.roles.length) {
                meta.appendChild(el('span', null, `${t.roles.length} 个角色`));
            }
            if (typeof t.unsupported_rate_final === 'number') {
                meta.appendChild(el('span', null, `无据 ${t.unsupported_rate_final}%`));
            }
            item.appendChild(meta);

            item.addEventListener('click', function () { select(t.run_id); });
            runList.appendChild(item);
        });
    }

    // ---------- 汇总指标 ----------
    function stat(key, value, tone) {
        const box = el('div', 'stat' + (tone ? ' ' + tone : ''));
        box.appendChild(el('div', 'k', key));
        box.appendChild(el('div', 'v', value));
        return box;
    }

    function renderStats(data) {
        const s = data.summary || {};
        const wrap = el('div', 'trace-stats');
        wrap.appendChild(stat('状态', statusLabel(data.status),
            data.status === 'completed' ? 'good' : 'bad'));
        wrap.appendChild(stat('步数', `${s.steps ?? '—'} / ${s.max_steps ?? '—'}`));
        wrap.appendChild(stat('工具调用', s.tool_calls ?? '—'));
        wrap.appendChild(stat('Token', s.total_tokens ?? '—'));
        wrap.appendChild(stat('成本', fmtCost(s.cost_usd)));
        wrap.appendChild(stat('耗时', s.duration_s != null ? `${s.duration_s}s` : '—'));

        // 有 Critic 参与才显示引用校验的结果
        if (typeof s.unsupported_rate_final === 'number') {
            wrap.appendChild(stat('无据结论率', `${s.unsupported_rate_final}%`,
                s.unsupported_rate_final === 0 ? 'good' : 'bad'));
            wrap.appendChild(stat('有据结论',
                `${s.claims_grounded ?? '—'} / ${s.claims_total ?? '—'}`));
            if (s.revisions) wrap.appendChild(stat('修订轮数', s.revisions));
        }
        if (s.repair_attempts) wrap.appendChild(stat('schema 自修复', s.repair_attempts));
        return wrap;
    }

    // ---------- 单步 ----------
    function tag(text, cls) {
        return el('span', 'tag ' + cls, text);
    }

    function renderCritic(step) {
        /* Critic 那一步的 observation 是一整份 JSON 判定报告。
           直接把 4000 字符的 JSON 摊在页面上没人看，拆成一条条结论。 */
        const body = el('div', 'step-body');
        body.appendChild(el('p', null, step.thought || ''));

        let report;
        try {
            report = JSON.parse(step.observation);
        } catch (e) {
            return body;
        }
        const list = el('ul', 'verdicts');
        (report.verdicts || []).forEach(function (v) {
            const li = el('li');
            li.appendChild(el('span', 'mark ' + (v.grounded ? 'ok' : 'no'),
                v.grounded ? '✓' : '✗'));
            const txt = el('div');
            txt.appendChild(el('div', null, v.claim));
            if (v.reasons && v.reasons.length) {
                txt.appendChild(el('span', 'why', v.reasons.join('；')));
            } else if (v.supporting_ids && v.supporting_ids.length) {
                txt.appendChild(el('span', 'why',
                    '出处 review_id: ' + v.supporting_ids.join(', ')));
            }
            li.appendChild(txt);
            list.appendChild(li);
        });
        if (list.childNodes.length) body.appendChild(list);
        return body;
    }

    function renderStep(step) {
        const failed = step.ok === false;
        const wrap = el('div', `step k-${step.kind}${failed ? ' failed' : ''}`);
        const card = el('div', 'step-card');

        // 头部
        const head = el('div', 'step-head');
        head.appendChild(el('span', 'step-n', `#${step.step}`));
        head.appendChild(tag(step.kind, 'tag-' + step.kind));
        if (step.role) head.appendChild(tag(step.role, 'tag-role'));
        if (step.tool) head.appendChild(el('span', 'kv', step.tool));
        if (step.repaired) head.appendChild(tag('schema 自修复', 'tag-repair'));

        const bits = [];
        if (step.latency_ms) bits.push(`${step.latency_ms} ms`);
        if (step.prompt_tokens || step.completion_tokens) {
            bits.push(`${step.prompt_tokens}+${step.completion_tokens} tok`);
        }
        if (step.cost_usd) bits.push(fmtCost(step.cost_usd));
        if (bits.length) head.appendChild(el('span', 'step-cost', bits.join(' · ')));
        card.appendChild(head);

        // 正文
        if (step.kind === 'critic') {
            card.appendChild(renderCritic(step));
        } else {
            const body = el('div', 'step-body');
            if (step.thought) body.appendChild(el('p', null, step.thought));
            if (step.tool_args) {
                body.appendChild(el('div', 'kv', JSON.stringify(step.tool_args)));
            }
            if (step.error) body.appendChild(el('div', 'err', step.error));
            if (step.observation) {
                // 工具输出可能很长，折叠起来，想看的人点开
                const det = el('details');
                det.appendChild(el('summary', null, '查看工具返回'));
                det.appendChild(el('pre', 'obs', step.observation));
                body.appendChild(det);
            }
            card.appendChild(body);
        }

        wrap.appendChild(card);
        return wrap;
    }

    // ---------- 右栏 ----------
    function renderDetail(data) {
        detail.replaceChildren();

        const head = el('div', 'trace-head');
        head.appendChild(el('h1', null, data.task || '(无任务描述)'));
        const sub = el('div', 'run-id');
        sub.textContent = `${data.run_id} · ${data.model || '未知模型'} · ${fmtTime(data.started_at)}`;
        head.appendChild(sub);
        head.appendChild(renderStats(data));
        detail.appendChild(head);

        if (data.answer) {
            const ans = el('div', 'step-card');
            const h = el('div', 'step-head');
            h.appendChild(tag('最终答案', 'tag-final'));
            ans.appendChild(h);
            const b = el('div', 'step-body');
            b.appendChild(el('p', null, data.answer));
            ans.appendChild(b);
            detail.appendChild(ans);
            detail.appendChild(el('div', null, ' '));
        }

        const timeline = el('div', 'timeline');
        (data.steps || []).forEach(function (s) { timeline.appendChild(renderStep(s)); });
        detail.appendChild(timeline);
        detail.scrollTop = 0;
    }

    function showMessage(text) {
        detail.replaceChildren();
        const box = el('div', 'trace-empty');
        box.appendChild(el('h3', null, text));
        detail.appendChild(box);
    }

    // ---------- 数据 ----------
    function select(runId) {
        activeId = runId;
        renderRunList();
        showMessage('加载中…');
        fetch(`/api/traces/${encodeURIComponent(runId)}`)
            .then(function (r) {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then(renderDetail)
            .catch(function (e) { showMessage(`轨迹加载失败：${e.message}`); });
    }

    function load() {
        fetch('/api/traces')
            .then(function (r) {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then(function (data) {
                traces = data.traces || [];
                renderRunList();
                if (traces.length && !activeId) select(traces[0].run_id);
            })
            .catch(function (e) {
                runList.replaceChildren(el('p', 'trace-hint', `列表加载失败：${e.message}`));
            });
    }

    if (refreshBtn) refreshBtn.addEventListener('click', load);
    load();
})();
