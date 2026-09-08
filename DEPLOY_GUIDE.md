# Streamlit Cloud 部署指南（screening-tool）

## 1. 推送本地提交到 GitHub

本地已生成 commit `03d59ef`，包含真实 13 变量 app、真实模型工件、更新 README。

### 方式 A：GitHub Desktop（推荐，无需命令行 PAT）
1. 打开 GitHub Desktop → 选择 `screening-tool` 仓库。
2. 确认右上角显示 **1 commit to push**。
3. 点击 **Push origin** 按钮。
4. 等待进度条完成。

### 方式 B：命令行 + PAT
如果你 prefer 命令行，需要一个 `repo` 权限的 GitHub Personal Access Token：

```bash
cd D:/1.20260902323文章修改/06_Data_and_Code/github_repo
# 临时将 remote URL 改成含 PAT 的形式（推送后建议改回普通 HTTPS）
git remote set-url origin https://<你的PAT>@github.com/community-sleep/screening-tool.git
git push origin main
git remote set-url origin https://github.com/community-sleep/screening-tool.git
```

> ⚠️ 安全提示：PAT 属于敏感凭证，用完立即在 GitHub Settings → Developer settings → Personal access tokens → Tokens (classic) 中**删除或轮换**。

## 2. 在 Streamlit Cloud 部署

1. 访问 <https://streamlit.io/cloud> 并用 GitHub 账号登录。
2. 点击 **New app**。
3. **Repository**：选择 `community-sleep/screening-tool`。
4. **Branch**：`main`。
5. **Main file path**：`app.py`。
6. **App URL**：设为 `https://sleep-risk-screening.streamlit.app`。
7. 点击 **Deploy**。

首次部署需 1–3 分钟安装依赖（`requirements.txt` 已包含 streamlit/xgboost/shap 等）。部署成功后，打开 URL 应能看到 13 变量输入界面，点击 **Predict sleep disorder risk** 后显示风险概率与三级分层。

## 3. 部署后检查清单

- [ ] GitHub 仓库最新 commit 为 `03d59ef`（可在 GitHub 网页确认）。
- [ ] Streamlit app URL 可访问：`https://sleep-risk-screening.streamlit.app`
- [ ] 输入示例：Age=70、Sex=Male、Current smoking=Yes、Depression=Yes、Fatigue=Yes、Hypertension=Yes、Diabetes=Yes，其余 No → 预测概率应接近 0.99，Risk category = High Risk。
- [ ] README 首页 metrics 与正文一致（Training AUC 0.922 / Internal 0.895 / External 0.802）。

## 4. 部署失败常见排查

- **App 启动失败 / ModuleNotFoundError**：检查 `requirements.txt` 是否被正确读取；可尝试在 Streamlit Cloud "Manage app" → "Reboot" 重新构建。
- **模型加载失败 / XGBoost version mismatch**：`models/xgboost_sleep_model.pkl` 是用 xgboost 2.x 保存的；确保 `requirements.txt` 中 `xgboost>=2.0`。
- **页面空白/404**：确认 Main file path 是 `app.py`（不是子目录路径）。
- **URL 与正文不一致**：正文目前写的是 "will be made freely available upon publication"，无需改 URL；如你希望改为 "available at https://..."，推送后按 P0 反向替换即可。

## 5. eFigure 1 截图

本地已生成 `D:/1.20260902323文章修改/05_Supplementary_Materials/02_eFigures/eFigure1/eFigure1_final_composite.png`，并已插入主稿补充材料（eFigure 1）。截图中的输入界面为 13 变量，预测输出显示 High Risk 99.5%，与正文 Table 4 三级分层一致。
