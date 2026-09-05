@echo off
REM ============================================================
REM  GitHub 发布脚本 — HCC Risk Calculator
REM  使用方法：双击运行本文件
REM ============================================================

setlocal

REM ---- 配置 ----
set "REPO_NAME=hcc-risk-calculator"
set "REPO_DESC=HCC risk calculator: SHAP-based ML web app for hepatocellular carcinoma risk prediction"
set "REPO_VISIBILITY=--public"
set "GIT_USER_NAME=HCC Risk Calculator"
set "GIT_USER_EMAIL=research@example.com"
set "LOCAL_DIR=%~dp0"

echo.
echo ============================================================
echo  GitHub 发布脚本
echo  仓库名: %REPO_NAME%
echo  仓库可见性: public
echo ============================================================
echo.

REM ---- 切换到脚本所在目录 ----
cd /d "%LOCAL_DIR%"
echo [1/6] 当前目录: %CD%
echo.

REM ---- 检查 git ----
where git >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 git，请先安装 Git for Windows
    echo 下载: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [2/6] git 已安装: 
git --version
echo.

REM ---- 检查 gh ----
where gh >nul 2>&1
if errorlevel 1 (
    echo [提示] 未在 PATH 中找到 gh，使用完整路径
    set "GH=C:\Program Files\GitHub CLI\gh.exe"
) else (
    set "GH=gh"
)
if not exist "%GH%" (
    echo [错误] 未找到 gh，请先安装 GitHub CLI
    echo 安装: winget install --id GitHub.cli -e --source winget
    pause
    exit /b 1
)
echo [3/6] gh 已安装:
"%GH%" --version
echo.

REM ---- 检查认证状态 ----
echo [4/6] 检查 GitHub 认证状态...
"%GH%" auth status >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  需要先完成 GitHub 认证
    echo ============================================================
    echo.
    echo  请选择认证方式:
    echo    [1] 浏览器认证 (推荐):    "%GH%" auth login --web
    echo    [2] PAT 认证:             "%GH%" auth login --with-token ^< token.txt
    echo.
    echo  关于 PAT 创建: GitHub → Settings → Developer settings
    echo    → Personal access tokens → Tokens (classic)
    echo    → Generate new token, 勾选 repo 权限, 复制 token
    echo.
    set /p AUTH_METHOD=请选择认证方式 [1/2]:
    if "%AUTH_METHOD%"=="1" (
        "%GH%" auth login --web
    ) else (
        echo.
        set /p GH_TOKEN=请粘贴你的 GitHub PAT 然后按回车:
        echo %GH_TOKEN% | "%GH%" auth login --with-token
    )
    echo.
)

REM ---- 再次检查认证 ----
"%GH%" auth status >nul 2>&1
if errorlevel 1 (
    echo [错误] 认证失败，请重试
    pause
    exit /b 1
)
echo    认证成功
echo.

REM ---- 配置本地 git 用户 (如果未配置) ----
git config user.name >nul 2>&1
if errorlevel 1 git config user.name "%GIT_USER_NAME%"
git config user.email >nul 2>&1
if errorlevel 1 git config user.email "%GIT_USER_EMAIL%"
echo [5/6] git 用户已配置: %GIT_USER_NAME% ^<%GIT_USER_EMAIL%^>
echo.

REM ---- 检查是否已初始化 ----
if not exist ".git" (
    echo 初始化本地 git 仓库...
    git init -b main
    git add -A
    git commit -m "Initial commit: HCC risk calculator app and training pipeline"
)
echo.

REM ---- 创建远程仓库并推送 ----
echo [6/6] 创建远程仓库并推送代码...
"%GH%" repo create %REPO_NAME% %REPO_VISIBILITY% --source=. --remote=origin --description="%REPO_DESC%" --push 2>&1
if errorlevel 1 (
    echo.
    echo [提示] 远程仓库可能已存在，尝试直接推送...
    git push -u origin main
)

echo.
echo ============================================================
echo  完成！
echo ============================================================
echo.
"%GH%" repo view %REPO_NAME% --web 2>nul
echo.
pause