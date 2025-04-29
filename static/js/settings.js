/**
 * 设置页面功能模块
 */

// 默认设置
const DEFAULT_SETTINGS = {
    general: {
        language: 'zh_CN',
        defaultModel: 'cnn',
        autoSave: true,
        confirmExit: true
    },
    appearance: {
        theme: 'light',
        fontSize: 14,
        sidebarPosition: 'left'
    },
    notifications: {
        enabled: true,
        analysisComplete: true,
        systemUpdates: true,
        adminNotifications: false,
        browserNotifications: true,
        inAppNotifications: true,
        emailNotifications: false
    },
    advanced: {
        sessionTimeout: 30
    }
};

// 全局变量
let currentSettings = {};
let settingsChanged = false;
let lastSavedSettings = {};

document.addEventListener('DOMContentLoaded', function() {
    // 初始化设置
    initSettings();
    
    // 表单提交处理
    initFormSubmitHandlers();
    
    // 添加主题预览按钮事件
    const previewThemeBtn = document.getElementById('previewThemeBtn');
    if (previewThemeBtn) {
        previewThemeBtn.addEventListener('click', previewTheme);
    }
    
    // 添加测试通知按钮事件
    const testNotificationBtn = document.getElementById('testNotificationBtn');
    if (testNotificationBtn) {
        testNotificationBtn.addEventListener('click', showNotificationPreview);
    }
    
    // 添加主通知切换事件
    const enableNotifications = document.getElementById('enableNotifications');
    if (enableNotifications) {
        enableNotifications.addEventListener('change', toggleNotificationOptions);
    }
    
    // 添加字体大小滑块事件
    const fontSizeRange = document.getElementById('fontSizeRange');
    if (fontSizeRange) {
        fontSizeRange.addEventListener('input', updateFontSizePreview);
    }
    
    // 添加会话超时滑块事件
    const sessionTimeoutRange = document.getElementById('sessionTimeoutRange');
    const sessionTimeout = document.getElementById('sessionTimeout');
    if (sessionTimeoutRange && sessionTimeout) {
        sessionTimeoutRange.addEventListener('input', function() {
            sessionTimeout.value = this.value;
        });
        sessionTimeout.addEventListener('change', function() {
            sessionTimeoutRange.value = this.value;
        });
    }
    
    // 添加导出设置按钮事件
    const exportSettingsBtn = document.getElementById('exportSettingsBtn');
    if (exportSettingsBtn) {
        exportSettingsBtn.addEventListener('click', exportSettings);
    }
    
    // 添加导入设置文件选择事件
    const importSettings = document.getElementById('importSettings');
    const settingsFileName = document.getElementById('settingsFileName');
    if (importSettings && settingsFileName) {
        importSettings.addEventListener('change', function() {
            if (this.files.length > 0) {
                settingsFileName.textContent = this.files[0].name;
                importSettingsFromFile(this.files[0]);
            } else {
                settingsFileName.textContent = '未选择文件';
            }
        });
    }
    
    // 添加清除缓存按钮事件
    const clearCacheBtn = document.getElementById('clearCacheBtn');
    if (clearCacheBtn) {
        clearCacheBtn.addEventListener('click', clearLocalCache);
    }
    
    // 添加重置设置按钮事件
    const resetSettingsBtn = document.getElementById('resetSettingsBtn');
    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener('click', function() {
            const resetModal = new bootstrap.Modal(document.getElementById('resetConfirmModal'));
            resetModal.show();
        });
    }
    
    // 添加确认重置按钮事件
    const confirmResetBtn = document.getElementById('confirmResetBtn');
    if (confirmResetBtn) {
        confirmResetBtn.addEventListener('click', resetAllSettings);
    }
    
    // 添加保存所有设置按钮事件
    const saveAllSettings = document.getElementById('saveAllSettings');
    if (saveAllSettings) {
        saveAllSettings.addEventListener('click', saveAllSettingsHandler);
    }
    
    // 监听所有设置变化
    monitorSettingsChanges();
    
    // 检查页面加载时的URL片段，以激活相应的标签
    checkUrlHash();
    
    // 添加标签切换事件，更新URL
    const tabs = document.querySelectorAll('.settings-tabs .list-group-item');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            window.location.hash = this.getAttribute('href');
        });
    });
    
    // 添加模型选择事件
    const modelOptions = document.querySelectorAll('input[name="defaultModel"]');
    modelOptions.forEach(option => {
        option.addEventListener('change', function() {
            if (this.checked) {
                document.getElementById('defaultModelSelect').value = this.value;
                settingsChanged = true;
                updateStatusBadge('generalSettingsStatus', 'unsaved');
            }
        });
    });
});

/**
 * 检查URL片段并激活相应标签
 */
function checkUrlHash() {
    const hash = window.location.hash;
    if (hash) {
        // 找到与哈希匹配的标签并激活它
        const targetTab = document.querySelector(`.settings-tabs .list-group-item[href="${hash}"]`);
        if (targetTab) {
            targetTab.click();
        }
    }
}

/**
 * 初始化设置
 */
async function initSettings() {
    try {
        showPageLoader();
        
        // 从API获取用户设置
        const response = await fetch('/api/settings');
        let settings;
        
        if (response.ok) {
            settings = await response.json();
        } else {
            console.warn('无法从服务器获取设置，使用默认设置');
            settings = { ...DEFAULT_SETTINGS };
        }
        
        // 合并默认设置和用户设置以确保所有字段都存在
        currentSettings = mergeSettings(DEFAULT_SETTINGS, settings);
        lastSavedSettings = JSON.parse(JSON.stringify(currentSettings));
        
        // 将设置应用到表单
        applySettingsToForms(currentSettings);
        
        // 应用初始UI状态
        applyInitialUIState();
        
    } catch (error) {
        console.error('初始化设置失败:', error);
        showToast('无法加载设置，使用默认设置', 'warning');
        
        // 使用默认设置
        currentSettings = { ...DEFAULT_SETTINGS };
        lastSavedSettings = { ...DEFAULT_SETTINGS };
        
        // 将默认设置应用到表单
        applySettingsToForms(currentSettings);
    } finally {
        hidePageLoader();
    }
}

/**
 * 合并默认设置和用户设置
 */
function mergeSettings(defaults, userSettings) {
    const result = { ...defaults };
    
    // 遍历默认设置的每个部分
    for (const section in defaults) {
        if (userSettings && userSettings[section]) {
            // 如果用户设置中有此部分，合并该部分的设置
            result[section] = { ...defaults[section], ...userSettings[section] };
        }
    }
    
    return result;
}

/**
 * 将设置应用到表单
 */
function applySettingsToForms(settings) {
    // 常规设置
    document.getElementById('languageSelect').value = settings.general.language;
    
    // 默认模型选择
    document.getElementById('defaultModelSelect').value = settings.general.defaultModel;
    const modelRadio = document.querySelector(`input[name="defaultModel"][value="${settings.general.defaultModel}"]`);
    if (modelRadio) {
        modelRadio.checked = true;
    }
    
    document.getElementById('autoSaveSwitch').checked = settings.general.autoSave;
    document.getElementById('confirmExitSwitch').checked = settings.general.confirmExit;
    
    // 外观设置
    const themeRadio = document.querySelector(`input[name="theme"][value="${settings.appearance.theme}"]`);
    if (themeRadio) {
        themeRadio.checked = true;
    }
    
    document.getElementById('fontSizeRange').value = settings.appearance.fontSize;
    updateFontSizePreview();
    
    const sidebarRadio = document.querySelector(`input[name="sidebarPosition"][value="${settings.appearance.sidebarPosition}"]`);
    if (sidebarRadio) {
        sidebarRadio.checked = true;
    }
    
    // 通知设置
    document.getElementById('enableNotifications').checked = settings.notifications.enabled;
    document.getElementById('analysisComplete').checked = settings.notifications.analysisComplete;
    document.getElementById('systemUpdates').checked = settings.notifications.systemUpdates;
    document.getElementById('adminNotifications').checked = settings.notifications.adminNotifications;
    document.getElementById('browserNotifications').checked = settings.notifications.browserNotifications;
    document.getElementById('inAppNotifications').checked = settings.notifications.inAppNotifications;
    document.getElementById('emailNotifications').checked = settings.notifications.emailNotifications;
    
    // 高级设置
    document.getElementById('sessionTimeout').value = settings.advanced.sessionTimeout;
    document.getElementById('sessionTimeoutRange').value = settings.advanced.sessionTimeout;
    
    // 应用通知选项状态
    toggleNotificationOptions();
}

/**
 * 应用初始UI状态
 */
function applyInitialUIState() {
    // 应用字体大小
    updateFontSizePreview();
    applyFontSize(currentSettings.appearance.fontSize);
    
    // 应用初始通知状态
    toggleNotificationOptions();
}

/**
 * 初始化表单提交处理器
 */
function initFormSubmitHandlers() {
    // 常规设置表单
    const generalSettingsForm = document.getElementById('generalSettingsForm');
    if (generalSettingsForm) {
        generalSettingsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            saveGeneralSettings();
        });
    }
    
    // 外观设置表单
    const appearanceSettingsForm = document.getElementById('appearanceSettingsForm');
    if (appearanceSettingsForm) {
        appearanceSettingsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            saveAppearanceSettings();
        });
    }
    
    // 通知设置表单
    const notificationSettingsForm = document.getElementById('notificationSettingsForm');
    if (notificationSettingsForm) {
        notificationSettingsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            saveNotificationSettings();
        });
    }
    
    // 高级设置表单
    const advancedSettingsForm = document.getElementById('advancedSettingsForm');
    if (advancedSettingsForm) {
        advancedSettingsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            saveAdvancedSettings();
        });
    }
}

/**
 * 监控设置变化
 */
function monitorSettingsChanges() {
    // 监听常规设置变化
    document.getElementById('languageSelect').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('generalSettingsStatus', 'unsaved');
    });
    
    document.getElementById('autoSaveSwitch').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('generalSettingsStatus', 'unsaved');
    });
    
    document.getElementById('confirmExitSwitch').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('generalSettingsStatus', 'unsaved');
    });
    
    // 监听外观设置变化
    document.querySelectorAll('input[name="theme"]').forEach(radio => {
        radio.addEventListener('change', () => {
            settingsChanged = true;
            updateStatusBadge('appearanceSettingsStatus', 'unsaved');
        });
    });
    
    document.getElementById('fontSizeRange').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('appearanceSettingsStatus', 'unsaved');
    });
    
    document.querySelectorAll('input[name="sidebarPosition"]').forEach(radio => {
        radio.addEventListener('change', () => {
            settingsChanged = true;
            updateStatusBadge('appearanceSettingsStatus', 'unsaved');
        });
    });
    
    // 监听通知设置变化
    document.getElementById('enableNotifications').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
    });
    
    document.getElementById('analysisComplete').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
    });
    
    document.getElementById('systemUpdates').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
    });
    
    document.getElementById('adminNotifications').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
    });
    
    document.getElementById('browserNotifications').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
    });
    
    document.getElementById('inAppNotifications').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
    });
    
    document.getElementById('emailNotifications').addEventListener('change', () => {
        settingsChanged = true;
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
    });
    
    // 监听高级设置变化
    document.getElementById('sessionTimeout').addEventListener('change', () => {
        settingsChanged = true;
    });
    
    document.getElementById('sessionTimeoutRange').addEventListener('change', () => {
        settingsChanged = true;
    });
}

/**
 * 更新状态徽章
 */
function updateStatusBadge(badgeId, status) {
    const badge = document.getElementById(badgeId);
    if (!badge) return;
    
    if (status === 'saved') {
        badge.innerHTML = '<i class="bi bi-check-circle-fill text-success"></i> 已保存';
    } else if (status === 'unsaved') {
        badge.innerHTML = '<i class="bi bi-exclamation-circle-fill text-warning"></i> 未保存';
    } else if (status === 'saving') {
        badge.innerHTML = '<div class="spinner-border spinner-border-sm text-primary" role="status"><span class="visually-hidden">保存中...</span></div> 保存中...';
    }
}

/**
 * 保存常规设置
 */
async function saveGeneralSettings() {
    try {
        // 显示保存状态
        updateStatusBadge('generalSettingsStatus', 'saving');
        toggleButtonLoading('generalSettingsForm', true);
        
        // 收集设置
        const settings = {
            general: {
                language: document.getElementById('languageSelect').value,
                defaultModel: document.getElementById('defaultModelSelect').value,
                autoSave: document.getElementById('autoSaveSwitch').checked,
                confirmExit: document.getElementById('confirmExitSwitch').checked
            }
        };
        
        // 调用API保存设置
        const success = await saveSettingsToAPI(settings);
        
        if (success) {
            // 更新当前设置
            currentSettings.general = { ...settings.general };
            lastSavedSettings.general = { ...settings.general };
            settingsChanged = false;
            
            // 显示成功状态
            updateStatusBadge('generalSettingsStatus', 'saved');
            showMessage('generalSettingsMessage', '常规设置已保存', 'success');
            
            // 如果语言设置已更改，可能需要刷新页面
            if (settings.general.language !== DEFAULT_SETTINGS.general.language) {
                showConfirmDialog('语言设置已更改', '需要刷新页面以使新的语言设置生效。是否立即刷新？', () => {
                    window.location.reload();
                });
            }
        } else {
            // 显示错误状态
            updateStatusBadge('generalSettingsStatus', 'unsaved');
            showMessage('generalSettingsMessage', '保存设置失败，请重试', 'danger');
        }
    } catch (error) {
        console.error('保存常规设置失败:', error);
        updateStatusBadge('generalSettingsStatus', 'unsaved');
        showMessage('generalSettingsMessage', '保存设置时发生错误', 'danger');
    } finally {
        toggleButtonLoading('generalSettingsForm', false);
    }
}

/**
 * 保存外观设置
 */
async function saveAppearanceSettings() {
    try {
        // 显示保存状态
        updateStatusBadge('appearanceSettingsStatus', 'saving');
        toggleButtonLoading('appearanceSettingsForm', true);
        
        // 获取主题选择
        const theme = document.querySelector('input[name="theme"]:checked').value;
        
        // 获取字体大小
        const fontSize = parseInt(document.getElementById('fontSizeRange').value);
        
        // 获取侧边栏位置
        const sidebarPosition = document.querySelector('input[name="sidebarPosition"]:checked').value;
        
        // 收集设置
        const settings = {
            appearance: {
                theme,
                fontSize,
                sidebarPosition
            }
        };
        
        // 调用API保存设置
        const success = await saveSettingsToAPI(settings);
        
        if (success) {
            // 更新当前设置
            currentSettings.appearance = { ...settings.appearance };
            lastSavedSettings.appearance = { ...settings.appearance };
            settingsChanged = false;
            
            // 应用字体大小
            applyFontSize(fontSize);
            
            // 显示成功状态
            updateStatusBadge('appearanceSettingsStatus', 'saved');
            showMessage('appearanceSettingsMessage', '外观设置已保存', 'success');
            
            // 如果主题设置已更改，提示应用新主题
            if (settings.appearance.theme !== DEFAULT_SETTINGS.appearance.theme) {
                showConfirmDialog('主题设置已更改', '需要刷新页面以应用新的主题。是否立即刷新？', () => {
                    window.location.reload();
                });
            }
        } else {
            // 显示错误状态
            updateStatusBadge('appearanceSettingsStatus', 'unsaved');
            showMessage('appearanceSettingsMessage', '保存设置失败，请重试', 'danger');
        }
    } catch (error) {
        console.error('保存外观设置失败:', error);
        updateStatusBadge('appearanceSettingsStatus', 'unsaved');
        showMessage('appearanceSettingsMessage', '保存设置时发生错误', 'danger');
    } finally {
        toggleButtonLoading('appearanceSettingsForm', false);
    }
}

/**
 * 保存通知设置
 */
async function saveNotificationSettings() {
    try {
        // 显示保存状态
        updateStatusBadge('notificationSettingsStatus', 'saving');
        toggleButtonLoading('notificationSettingsForm', true);
        
        // 收集设置
        const settings = {
            notifications: {
                enabled: document.getElementById('enableNotifications').checked,
                analysisComplete: document.getElementById('analysisComplete').checked,
                systemUpdates: document.getElementById('systemUpdates').checked,
                adminNotifications: document.getElementById('adminNotifications').checked,
                browserNotifications: document.getElementById('browserNotifications').checked,
                inAppNotifications: document.getElementById('inAppNotifications').checked,
                emailNotifications: document.getElementById('emailNotifications').checked
            }
        };
        
        // 调用API保存设置
        const success = await saveSettingsToAPI(settings);
        
        if (success) {
            // 更新当前设置
            currentSettings.notifications = { ...settings.notifications };
            lastSavedSettings.notifications = { ...settings.notifications };
            settingsChanged = false;
            
            // 显示成功状态
            updateStatusBadge('notificationSettingsStatus', 'saved');
            showMessage('notificationSettingsMessage', '通知设置已保存', 'success');
            
            // 检查浏览器通知权限
            if (settings.notifications.enabled && settings.notifications.browserNotifications) {
                checkNotificationPermission();
            }
        } else {
            // 显示错误状态
            updateStatusBadge('notificationSettingsStatus', 'unsaved');
            showMessage('notificationSettingsMessage', '保存设置失败，请重试', 'danger');
        }
    } catch (error) {
        console.error('保存通知设置失败:', error);
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
        showMessage('notificationSettingsMessage', '保存设置时发生错误', 'danger');
    } finally {
        toggleButtonLoading('notificationSettingsForm', false);
    }
}

/**
 * 保存高级设置
 */
async function saveAdvancedSettings() {
    try {
        // 显示保存状态
        toggleButtonLoading('advancedSettingsForm', true);
        
        // 收集设置
        const settings = {
            advanced: {
                sessionTimeout: parseInt(document.getElementById('sessionTimeout').value)
            }
        };
        
        // 调用API保存设置
        const success = await saveSettingsToAPI(settings);
        
        if (success) {
            // 更新当前设置
            currentSettings.advanced = { ...settings.advanced };
            lastSavedSettings.advanced = { ...settings.advanced };
            settingsChanged = false;
            
            // 显示成功状态
            showMessage('advancedSettingsMessage', '高级设置已保存', 'success');
        } else {
            // 显示错误状态
            showMessage('advancedSettingsMessage', '保存设置失败，请重试', 'danger');
        }
    } catch (error) {
        console.error('保存高级设置失败:', error);
        showMessage('advancedSettingsMessage', '保存设置时发生错误', 'danger');
    } finally {
        toggleButtonLoading('advancedSettingsForm', false);
    }
}

/**
 * 保存所有设置
 */
async function saveAllSettingsHandler() {
    try {
        // 显示加载状态
        const saveAllBtn = document.getElementById('saveAllSettings');
        const originalText = saveAllBtn.innerHTML;
        saveAllBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 保存中...';
        saveAllBtn.disabled = true;
        
        // 更新所有状态徽章
        updateStatusBadge('generalSettingsStatus', 'saving');
        updateStatusBadge('appearanceSettingsStatus', 'saving');
        updateStatusBadge('notificationSettingsStatus', 'saving');
        
        // 收集所有设置
        const settings = {
            general: {
                language: document.getElementById('languageSelect').value,
                defaultModel: document.getElementById('defaultModelSelect').value,
                autoSave: document.getElementById('autoSaveSwitch').checked,
                confirmExit: document.getElementById('confirmExitSwitch').checked
            },
            appearance: {
                theme: document.querySelector('input[name="theme"]:checked').value,
                fontSize: parseInt(document.getElementById('fontSizeRange').value),
                sidebarPosition: document.querySelector('input[name="sidebarPosition"]:checked').value
            },
            notifications: {
                enabled: document.getElementById('enableNotifications').checked,
                analysisComplete: document.getElementById('analysisComplete').checked,
                systemUpdates: document.getElementById('systemUpdates').checked,
                adminNotifications: document.getElementById('adminNotifications').checked,
                browserNotifications: document.getElementById('browserNotifications').checked,
                inAppNotifications: document.getElementById('inAppNotifications').checked,
                emailNotifications: document.getElementById('emailNotifications').checked
            },
            advanced: {
                sessionTimeout: parseInt(document.getElementById('sessionTimeout').value)
            }
        };
        
        // 调用API保存设置
        const success = await saveSettingsToAPI(settings);
        
        if (success) {
            // 更新当前设置
            currentSettings = { ...settings };
            lastSavedSettings = { ...settings };
            settingsChanged = false;
            
            // 更新所有状态徽章
            updateStatusBadge('generalSettingsStatus', 'saved');
            updateStatusBadge('appearanceSettingsStatus', 'saved');
            updateStatusBadge('notificationSettingsStatus', 'saved');
            
            // 显示成功消息
            showToast('所有设置已成功保存', 'success');
            
            // 检查是否需要刷新页面
            const themeChanged = settings.appearance.theme !== DEFAULT_SETTINGS.appearance.theme;
            const languageChanged = settings.general.language !== DEFAULT_SETTINGS.general.language;
            
            if (themeChanged || languageChanged) {
                showConfirmDialog('设置已更改', '某些设置需要刷新页面才能生效。是否立即刷新？', () => {
                    window.location.reload();
                });
            }
            
            // 应用字体大小
            applyFontSize(settings.appearance.fontSize);
        } else {
            // 显示错误状态
            updateStatusBadge('generalSettingsStatus', 'unsaved');
            updateStatusBadge('appearanceSettingsStatus', 'unsaved');
            updateStatusBadge('notificationSettingsStatus', 'unsaved');
            
            // 显示错误消息
            showToast('保存设置失败，请重试', 'danger');
        }
    } catch (error) {
        console.error('保存所有设置失败:', error);
        // 显示错误状态
        updateStatusBadge('generalSettingsStatus', 'unsaved');
        updateStatusBadge('appearanceSettingsStatus', 'unsaved');
        updateStatusBadge('notificationSettingsStatus', 'unsaved');
        
        showToast('保存设置时发生错误', 'danger');
    } finally {
        // 恢复按钮状态
        const saveAllBtn = document.getElementById('saveAllSettings');
        saveAllBtn.innerHTML = '<i class="bi bi-save me-2"></i> 保存所有设置';
        saveAllBtn.disabled = false;
    }
}

/**
 * 保存设置到API
 */
async function saveSettingsToAPI(settings) {
    try {
        // 模拟API延迟
        await new Promise(resolve => setTimeout(resolve, 600));
        
        // 实际API调用
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });
        
        if (response.ok) {
            // 将设置保存在本地存储中，作为备份
            try {
                localStorage.setItem('userSettings', JSON.stringify({
                    ...JSON.parse(localStorage.getItem('userSettings') || '{}'),
                    ...settings
                }));
            } catch (storageError) {
                console.warn('无法保存设置到本地存储:', storageError);
            }
            
            return true;
        }
        
        return false;
    } catch (error) {
        console.error('保存设置到API失败:', error);
        
        // 将设置保存在本地存储中，作为备份
        try {
            localStorage.setItem('userSettings', JSON.stringify({
                ...JSON.parse(localStorage.getItem('userSettings') || '{}'),
                ...settings
            }));
            
            // 在开发模式下，允许本地存储成功也算成功
            if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                return true;
            }
        } catch (storageError) {
            console.warn('无法保存设置到本地存储:', storageError);
        }
        
        return false;
    }
}

/**
 * 切换通知选项的可用状态
 */
function toggleNotificationOptions() {
    const enabled = document.getElementById('enableNotifications').checked;
    const notificationOptions = document.querySelector('.notification-options');
    
    if (enabled) {
        notificationOptions.classList.remove('disabled');
        document.querySelectorAll('.notification-options input').forEach(input => {
            input.disabled = false;
        });
    } else {
        notificationOptions.classList.add('disabled');
        document.querySelectorAll('.notification-options input').forEach(input => {
            input.disabled = true;
        });
    }
}

/**
 * 更新字体大小预览
 */
function updateFontSizePreview() {
    const fontSize = document.getElementById('fontSizeRange').value;
    const fontSizeValue = document.getElementById('fontSizeValue');
    const fontSizeSample = document.querySelector('.font-size-sample');
    
    if (fontSizeValue) {
        fontSizeValue.textContent = `${fontSize}px`;
    }
    
    if (fontSizeSample) {
        fontSizeSample.style.fontSize = `${fontSize}px`;
    }
}

/**
 * 应用字体大小
 */
function applyFontSize(size) {
    // 这里只是示例，实际应用可能需要更复杂的实现
    document.documentElement.style.setProperty('--base-font-size', `${size}px`);
}

/**
 * 预览主题
 */
function previewTheme() {
    const selectedTheme = document.querySelector('input[name="theme"]:checked').value;
    const previewContainer = document.getElementById('themePreviewContainer');
    
    // 移除现有的主题类
    previewContainer.classList.remove('light', 'dark');
    
    // 应用选中的主题
    if (selectedTheme === 'auto') {
        // 检查系统主题
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        previewContainer.classList.add(prefersDark ? 'dark' : 'light');
    } else {
        previewContainer.classList.add(selectedTheme);
    }
    
    // 显示预览模态框
    const modal = new bootstrap.Modal(document.getElementById('themePreviewModal'));
    modal.show();
}

/**
 * 显示通知预览
 */
function showNotificationPreview() {
    // 显示预览模态框
    const modal = new bootstrap.Modal(document.getElementById('notificationPreviewModal'));
    modal.show();
    
    // 检查浏览器通知权限
    checkNotificationPermission();
}

/**
 * 检查浏览器通知权限
 */
function checkNotificationPermission() {
    if (!('Notification' in window)) {
        console.warn('此浏览器不支持桌面通知');
        return;
    }
    
    if (Notification.permission !== 'granted') {
        showToast('请允许浏览器通知以接收实时提醒', 'info', 5000);
        
        // 请求权限
        Notification.requestPermission();
    }
}

/**
 * 清除本地缓存
 */
async function clearLocalCache() {
    try {
        // 显示确认对话框
        if (!confirm('确定要清除本地缓存吗？这将删除所有本地存储的数据和设置。')) {
            return;
        }
        
        const btn = document.getElementById('clearCacheBtn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 清除中...';
        btn.disabled = true;
        
        // 清除localStorage (保留一些关键数据)
        const authToken = localStorage.getItem('authToken');
        localStorage.clear();
        if (authToken) {
            localStorage.setItem('authToken', authToken);
        }
        
        // 清除sessionStorage
        sessionStorage.clear();
        
        // 清除所有cookies
        document.cookie.split(';').forEach(cookie => {
            const eqPos = cookie.indexOf('=');
            const name = eqPos > -1 ? cookie.substr(0, eqPos).trim() : cookie.trim();
            document.cookie = name + '=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';
        });
        
        // 显示成功消息
        showToast('本地缓存已清除', 'success');
        
        // 恢复按钮状态
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
            
            // 提示刷新页面
            showConfirmDialog('缓存已清除', '建议刷新页面以应用更改。是否立即刷新？', () => {
                window.location.reload();
            });
        }, 1000);
        
    } catch (error) {
        console.error('清除缓存失败:', error);
        showToast('清除缓存失败', 'danger');
        
        // 恢复按钮状态
        const btn = document.getElementById('clearCacheBtn');
        if (btn) {
            btn.innerHTML = '<i class="bi bi-trash me-2"></i>清除本地缓存';
            btn.disabled = false;
        }
    }
}

/**
 * 导出设置
 */
function exportSettings() {
    try {
        // 准备导出数据
        const exportData = {
            settings: currentSettings,
            exportDate: new Date().toISOString(),
            version: '1.0'
        };
        
        // 转换为JSON字符串
        const jsonString = JSON.stringify(exportData, null, 2);
        
        // 创建Blob
        const blob = new Blob([jsonString], { type: 'application/json' });
        
        // 创建下载链接
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        
        // 设置文件名
        const date = new Date().toISOString().split('T')[0];
        link.download = `settings_export_${date}.json`;
        
        // 触发下载
        document.body.appendChild(link);
        link.click();
        
        // 清理
        URL.revokeObjectURL(url);
        document.body.removeChild(link);
        
        // 显示成功消息
        showToast('设置已成功导出', 'success');
        
    } catch (error) {
        console.error('导出设置失败:', error);
        showToast('导出设置失败', 'danger');
    }
}

/**
 * 从文件导入设置
 */
function importSettingsFromFile(file) {
    try {
        const reader = new FileReader();
        
        reader.onload = function(event) {
            try {
                // 解析JSON
                const importData = JSON.parse(event.target.result);
                
                // 验证导入的数据
                if (!importData.settings) {
                    throw new Error('无效的设置文件');
                }
                
                // 显示确认对话框
                showConfirmDialog('导入设置', '确定要导入这些设置吗？当前设置将被覆盖。', () => {
                    // 合并设置，确保所有必要字段都存在
                    const mergedSettings = mergeSettings(DEFAULT_SETTINGS, importData.settings);
                    
                    // 更新当前设置
                    currentSettings = { ...mergedSettings };
                    
                    // 应用设置到表单
                    applySettingsToForms(currentSettings);
                    
                    // 保存设置到API
                    saveSettingsToAPI(currentSettings)
                        .then(success => {
                            if (success) {
                                showToast('设置已成功导入', 'success');
                                
                                // 提示刷新页面
                                showConfirmDialog('设置已导入', '建议刷新页面以应用新设置。是否立即刷新？', () => {
                                    window.location.reload();
                                });
                            } else {
                                showToast('导入的设置无法保存到服务器', 'warning');
                            }
                        });
                });
                
            } catch (parseError) {
                console.error('解析导入文件失败:', parseError);
                showToast('无效的设置文件', 'danger');
                
                // 重置文件输入
                document.getElementById('importSettings').value = '';
                document.getElementById('settingsFileName').textContent = '未选择文件';
            }
        };
        
        reader.readAsText(file);
        
    } catch (error) {
        console.error('导入设置失败:', error);
        showToast('导入设置失败', 'danger');
        
        // 重置文件输入
        document.getElementById('importSettings').value = '';
        document.getElementById('settingsFileName').textContent = '未选择文件';
    }
}

/**
 * 重置所有设置
 */
function resetAllSettings() {
    try {
        // 更新当前设置为默认设置
        currentSettings = { ...DEFAULT_SETTINGS };
        
        // 应用设置到表单
        applySettingsToForms(currentSettings);
        
        // 保存设置到API
        saveSettingsToAPI(currentSettings)
            .then(success => {
                // 隐藏确认模态框
                const modal = bootstrap.Modal.getInstance(document.getElementById('resetConfirmModal'));
                modal.hide();
                
                if (success) {
                    // 更新所有状态徽章
                    updateStatusBadge('generalSettingsStatus', 'saved');
                    updateStatusBadge('appearanceSettingsStatus', 'saved');
                    updateStatusBadge('notificationSettingsStatus', 'saved');
                    
                    showToast('所有设置已重置为默认值', 'success');
                    
                    // 提示刷新页面
                    showConfirmDialog('设置已重置', '建议刷新页面以应用默认设置。是否立即刷新？', () => {
                        window.location.reload();
                    });
                } else {
                    showToast('无法保存默认设置到服务器', 'warning');
                }
            });
        
    } catch (error) {
        console.error('重置设置失败:', error);
        showToast('重置设置失败', 'danger');
    }
}

/**
 * 切换按钮加载状态
 */
function toggleButtonLoading(formId, isLoading) {
    const form = document.getElementById(formId);
    if (!form) return;
    
    const submitBtn = form.querySelector('button[type="submit"]');
    if (!submitBtn) return;
    
    if (isLoading) {
        submitBtn.querySelector('.btn-text').classList.add('d-none');
        submitBtn.querySelector('.btn-spinner').classList.remove('d-none');
        submitBtn.disabled = true;
    } else {
        submitBtn.querySelector('.btn-text').classList.remove('d-none');
        submitBtn.querySelector('.btn-spinner').classList.add('d-none');
        submitBtn.disabled = false;
    }
}

/**
 * 显示消息
 */
function showMessage(elementId, message, type) {
    const messageElement = document.getElementById(elementId);
    if (!messageElement) return;
    
    // 设置消息内容和类型
    messageElement.textContent = message;
    messageElement.className = `alert alert-${type}`;
    messageElement.classList.remove('d-none');
    
    // 添加动画
    messageElement.classList.add('alert-animated');
    
    // 如果是成功消息，3秒后自动隐藏
    if (type === 'success') {
        setTimeout(() => {
            messageElement.classList.add('alert-fade');
            setTimeout(() => {
                messageElement.classList.add('d-none');
                messageElement.classList.remove('alert-fade', 'alert-animated');
            }, 300);
        }, 3000);
    }
}

/**
 * 显示确认对话框
 */
function showConfirmDialog(title, message, confirmCallback) {
    // 创建自定义确认模态框
    const modalId = 'dynamicConfirmModal';
    
    // 检查是否已存在模态框
    let modal = document.getElementById(modalId);
    
    if (!modal) {
        // 创建模态框HTML
        const modalHTML = `
            <div class="modal fade" id="${modalId}" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${title}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p>${message}</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                            <button type="button" class="btn btn-primary" id="${modalId}ConfirmBtn">确认</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 将模态框添加到页面
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        modal = document.getElementById(modalId);
        
        // 添加确认按钮事件
        document.getElementById(`${modalId}ConfirmBtn`).addEventListener('click', function() {
            const modalInstance = bootstrap.Modal.getInstance(modal);
            modalInstance.hide();
            confirmCallback();
        });
    } else {
        // 更新现有模态框内容
        modal.querySelector('.modal-title').textContent = title;
        modal.querySelector('.modal-body p').textContent = message;
        
        // 更新确认按钮事件
        const confirmBtn = document.getElementById(`${modalId}ConfirmBtn`);
        const newConfirmBtn = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
        newConfirmBtn.addEventListener('click', function() {
            const modalInstance = bootstrap.Modal.getInstance(modal);
            modalInstance.hide();
            confirmCallback();
        });
    }
    
    // 显示模态框
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * 显示通知提示
 */
function showToast(message, type = 'info', duration = 3000) {
    // 创建toast容器（如果不存在）
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // 创建唯一ID
    const toastId = `toast_${Date.now()}`;
    
    // 确定图标
    let icon = 'info-circle';
    if (type === 'success') icon = 'check-circle';
    if (type === 'danger') icon = 'exclamation-circle';
    if (type === 'warning') icon = 'exclamation-triangle';
    
    // 创建toast元素
    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center border-0 bg-${type}" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="bi bi-${icon} me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    // 添加到容器
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    
    // 获取toast元素
    const toastElement = document.getElementById(toastId);
    
    // 初始化toast
    const toast = new bootstrap.Toast(toastElement, {
        animation: true,
        autohide: true,
        delay: duration
    });
    
    // 显示toast
    toast.show();
    
    // 在隐藏后移除元素
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

/**
 * 显示页面加载器
 */
function showPageLoader() {
    // 检查是否已存在加载器
    let loader = document.getElementById('pageLoader');
    if (!loader) {
        // 创建加载器
        loader = document.createElement('div');
        loader.id = 'pageLoader';
        loader.className = 'page-loader';
        loader.innerHTML = `
            <div class="page-loader-content">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">加载中...</span>
                </div>
                <div class="mt-2">加载设置中...</div>
            </div>
        `;
        document.body.appendChild(loader);
    } else {
        loader.style.display = 'flex';
    }
}

/**
 * 隐藏页面加载器
 */
function hidePageLoader() {
    const loader = document.getElementById('pageLoader');
    if (loader) {
        // 添加淡出动画
        loader.classList.add('fade-out');
        
        // 动画结束后移除
        setTimeout(() => {
            loader.remove();
        }, 300);
    }
}

// 添加页面加载器样式
document.head.insertAdjacentHTML('beforeend', `
<style>
    .page-loader {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 2000;
        animation: fadeIn 0.3s ease-in-out;
    }
    
    .page-loader-content {
        text-align: center;
    }
    
    .fade-out {
        animation: fadeOut 0.3s ease-in-out forwards;
    }
    
    /* 通知样式 */
    .notification-options.disabled {
        opacity: 0.6;
        pointer-events: none;
    }
    
    /* 动画 */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    
    /* 确保toast在模态框之上 */
    .toast-container {
        z-index: 1080;
    }
    
    /* Toast样式增强 */
    .toast {
        opacity: 0;
        transition: opacity 0.3s ease-in-out, transform 0.3s ease-in-out;
        transform: translateY(20px);
    }
    
    .toast.show {
        opacity: 1;
        transform: translateY(0);
    }
</style>
`);

// 添加页面离开前确认
window.addEventListener('beforeunload', function(e) {
    if (settingsChanged && currentSettings.general?.confirmExit) {
        const message = '您有未保存的设置更改，确定要离开吗？';
        e.returnValue = message;
        return message;
    }
});
