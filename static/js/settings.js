/**
 * 设置页面功能
 */

document.addEventListener('DOMContentLoaded', function () {
    // 使用auth.js中的用户信息，而不是从设置中获取
    if (typeof fetchUserInfo === 'function') {
        fetchUserInfo();
    }

    // 加载设置
    loadSettings();

    // 字体大小滑块事件
    const fontSizeRange = document.getElementById('fontSizeRange');
    if (fontSizeRange) {
        fontSizeRange.addEventListener('input', function () {
            document.getElementById('fontSizeValue').textContent = this.value + 'px';
        });
    }

    // 表单提交处理
    const generalForm = document.getElementById('generalSettingsForm');
    if (generalForm) {
        generalForm.addEventListener('submit', function (e) {
            e.preventDefault();
            saveGeneralSettings();
        });
    }

    const appearanceForm = document.getElementById('appearanceSettingsForm');
    if (appearanceForm) {
        appearanceForm.addEventListener('submit', function (e) {
            e.preventDefault();
            saveAppearanceSettings();
        });
    }

    const notificationForm = document.getElementById('notificationSettingsForm');
    if (notificationForm) {
        notificationForm.addEventListener('submit', function (e) {
            e.preventDefault();
            saveNotificationSettings();
        });
    }

    const advancedForm = document.getElementById('advancedSettingsForm');
    if (advancedForm) {
        advancedForm.addEventListener('submit', function (e) {
            e.preventDefault();
            saveAdvancedSettings();
        });
    }

    // 通知开关处理
    const enableNotifications = document.getElementById('enableNotifications');
    if (enableNotifications) {
        enableNotifications.addEventListener('change', function () {
            toggleNotificationOptions(this.checked);
        });
    }

    // 清除缓存按钮
    const clearCacheBtn = document.getElementById('clearCacheBtn');
    if (clearCacheBtn) {
        clearCacheBtn.addEventListener('click', function () {
            clearLocalCache();
        });
    }

    // 重置设置按钮
    const resetSettingsBtn = document.getElementById('resetSettingsBtn');
    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener('click', function () {
            if (confirm('确定要重置所有设置吗？此操作无法撤销。')) {
                resetAllSettings();
            }
        });
    }

    // 导出设置按钮
    const exportSettingsBtn = document.getElementById('exportSettingsBtn');
    if (exportSettingsBtn) {
        exportSettingsBtn.addEventListener('click', function () {
            exportSettings();
        });
    }

    // 导入设置文件
    const importSettings = document.getElementById('importSettings');
    if (importSettings) {
        importSettings.addEventListener('change', function (e) {
            importSettingsFromFile(e.target.files[0]);
        });
    }
});

/**
 * 加载用户设置
 */
async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        if (response.ok) {
            const data = await response.json();

            // 应用设置到表单
            if (data.theme) {
                const themeRadio = document.querySelector(`input[name="theme"][value="${data.theme}"]`);
                if (themeRadio) themeRadio.checked = true;
            }

            if (data.language) {
                const langSelect = document.getElementById('languageSelect');
                if (langSelect) langSelect.value = data.language;
            }

            if (data.sidebar_position) {
                const sidebarPosition = document.querySelector(`input[name="sidebarPosition"][value="${data.sidebar_position}"]`);
                if (sidebarPosition) sidebarPosition.checked = true;
            }

            // 其他设置
            const autoSave = document.getElementById('autoSaveSwitch');
            if (autoSave) autoSave.checked = data.auto_save !== false;

            const confirmExit = document.getElementById('confirmExitSwitch');
            if (confirmExit) confirmExit.checked = data.confirm_exit !== false;

            const notificationsEnabled = document.getElementById('enableNotifications');
            if (notificationsEnabled) {
                notificationsEnabled.checked = data.notifications_enabled !== false;
                toggleNotificationOptions(notificationsEnabled.checked);
            }

            const analysisComplete = document.getElementById('analysisComplete');
            if (analysisComplete) analysisComplete.checked = data.analysis_complete_notification !== false;

            const systemUpdates = document.getElementById('systemUpdates');
            if (systemUpdates) systemUpdates.checked = data.system_updates_notification !== false;

            const adminNotifications = document.getElementById('adminNotifications');
            if (adminNotifications) adminNotifications.checked = data.admin_notification === true;

            const sessionTimeout = document.getElementById('sessionTimeout');
            if (sessionTimeout) sessionTimeout.value = data.session_timeout || 30;

            const fontSizeRange = document.getElementById('fontSizeRange');
            if (fontSizeRange) {
                fontSizeRange.value = data.font_size || 14;
                document.getElementById('fontSizeValue').textContent = (data.font_size || 14) + 'px';
            }

            // 不再从设置API获取用户信息
            // 而是依赖auth.js中的用户信息加载机制
        } else {
            console.error('加载设置失败');
        }
    } catch (error) {
        console.error('获取设置出错:', error);
    }
}

/**
 * 保存常规设置
 */
async function saveGeneralSettings() {
    const settings = {
        language: document.getElementById('languageSelect').value,
        default_model: document.getElementById('defaultModelSelect').value,
        auto_save: document.getElementById('autoSaveSwitch').checked,
        confirm_exit: document.getElementById('confirmExitSwitch').checked
    };

    await saveSettings(settings, 'generalSettingsMessage');
}

/**
 * 保存界面外观设置
 */
async function saveAppearanceSettings() {
    const settings = {
        theme: document.querySelector('input[name="theme"]:checked').value,
        font_size: parseInt(document.getElementById('fontSizeRange').value),
        sidebar_position: document.querySelector('input[name="sidebarPosition"]:checked').value
    };

    await saveSettings(settings, 'appearanceSettingsMessage');
}

/**
 * 保存通知设置
 */
async function saveNotificationSettings() {
    const notificationsEnabled = document.getElementById('enableNotifications').checked;

    const settings = {
        notifications_enabled: notificationsEnabled,
        analysis_complete_notification: document.getElementById('analysisComplete').checked,
        system_updates_notification: document.getElementById('systemUpdates').checked,
        admin_notification: document.getElementById('adminNotifications').checked
    };

    await saveSettings(settings, 'notificationSettingsMessage');
}

/**
 * 保存高级设置
 */
async function saveAdvancedSettings() {
    const settings = {
        session_timeout: parseInt(document.getElementById('sessionTimeout').value)
    };

    await saveSettings(settings, 'advancedSettingsMessage');
}

/**
 * 保存设置到服务器
 * @param {Object} settings - 设置对象
 * @param {string} messageElementId - 消息显示元素ID
 */
async function saveSettings(settings, messageElementId) {
    try {
        const response = await fetch('/api/settings/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest' // 添加标识符确保服务器识别为AJAX请求
            },
            body: JSON.stringify(settings)
        });

        const data = await response.json();

        if (response.ok) {
            showMessage(messageElementId, '设置已保存', 'success');

            // 存储到本地存储，作为备份
            try {
                localStorage.setItem('userSettings', JSON.stringify(
                    {...JSON.parse(localStorage.getItem('userSettings') || '{}'), ...settings}
                ));
            } catch (e) {
                console.warn('无法保存设置到本地存储:', e);
            }
        } else {
            showMessage(messageElementId, data.error || '保存设置失败', 'danger');
        }
    } catch (error) {
        console.error('保存设置出错:', error);
        showMessage(messageElementId, '保存设置时发生错误', 'danger');
    }
}

/**
 * 切换通知选项的可用状态
 * @param {boolean} enabled - 是否启用通知
 */
function toggleNotificationOptions(enabled) {
    const options = document.querySelectorAll('.notification-options input');
    options.forEach(input => {
        input.disabled = !enabled;
    });
}

/**
 * 清除本地缓存
 */
function clearLocalCache() {
    try {
        localStorage.clear();
        sessionStorage.clear();
        showMessage('advancedSettingsMessage', '本地缓存已清除', 'success');
    } catch (error) {
        console.error('清除缓存出错:', error);
        showMessage('advancedSettingsMessage', '清除缓存时发生错误', 'danger');
    }
}

/**
 * 重置所有设置
 */
async function resetAllSettings() {
    try {
        // 调用API重置设置
        const response = await fetch('/api/settings/reset', {
            method: 'POST'
        });

        if (response.ok) {
            showMessage('advancedSettingsMessage', '所有设置已重置为默认值', 'success');
            // 重新加载页面以应用默认设置
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } else {
            showMessage('advancedSettingsMessage', '重置设置失败', 'danger');
        }
    } catch (error) {
        console.error('重置设置出错:', error);
        showMessage('advancedSettingsMessage', '重置设置时发生错误', 'danger');
    }
}

/**
 * 导出设置
 */
function exportSettings() {
    try {
        fetch('/api/settings')
            .then(response => response.json())
            .then(data => {
                // 创建下载链接
                const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'settings_' + new Date().toISOString().split('T')[0] + '.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                showMessage('advancedSettingsMessage', '设置已成功导出', 'success');
            });
    } catch (error) {
        console.error('导出设置出错:', error);
        showMessage('advancedSettingsMessage', '导出设置时发生错误', 'danger');
    }
}

/**
 * 从文件导入设置
 * @param {File} file - 设置文件
 */
function importSettingsFromFile(file) {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async function (e) {
        try {
            const settings = JSON.parse(e.target.result);

            // 调用API导入设置
            const response = await fetch('/api/settings/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });

            if (response.ok) {
                showMessage('advancedSettingsMessage', '设置已成功导入', 'success');
                // 重新加载页面以应用新设置
                setTimeout(() => {
                    window.location.reload();
                }, 1000);
            } else {
                showMessage('advancedSettingsMessage', '导入设置失败', 'danger');
            }
        } catch (error) {
            console.error('解析设置文件出错:', error);
            showMessage('advancedSettingsMessage', '无效的设置文件', 'danger');
        }
    };
    reader.readAsText(file);
}

/**
 * 显示消息
 * @param {string} elementId - 消息元素ID
 * @param {string} message - 显示的消息
 * @param {string} type - 消息类型 (success, danger, warning, info)
 */
function showMessage(elementId, message, type) {
    const messageElement = document.getElementById(elementId);
    if (messageElement) {
        messageElement.textContent = message;
        messageElement.className = `alert alert-${type}`;
        messageElement.classList.remove('d-none');

        // 成功消息3秒后自动隐藏
        if (type === 'success') {
            setTimeout(() => {
                messageElement.classList.add('d-none');
            }, 3000);
        }
    }
}
