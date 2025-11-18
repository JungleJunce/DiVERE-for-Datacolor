# FilmCCSoftware 架构设计

本文档旨在说明 FilmCCSoftware 的核心架构设计，尤其是在 ApplicationContext 重构之后。它将作为未来功能开发和维护的指导。

## 核心思想：单向数据流

我们建立的核心原则是单向数据流，这确保了应用状态的可预测性和易于调试。流程如下：

**UI 操作 → ApplicationContext 处理 → 状态变更 → UI 响应**

## 架构图 (Mermaid.js)

```mermaid
graph TD
    subgraph UI Layer
        A[MainWindow] -- User Action (e.g., Open Image) --> C
        B[ParameterPanel] -- User Action (e.g., Change Gamma) --> C
        A -- Creates & Owns --> B
        A -- Creates & Owns --> PreviewWidget
        A -- Owns --> C
    end

    subgraph Core Layer
        C[ApplicationContext]
        D[TheEnlarger]
        E[FilmPipelineProcessor]
        F[ImageManager]
        G[ColorSpaceManager]
        H[PresetManager]
    end

    subgraph Data Flow
        C -- Manages State (current_image, current_params) --> C
        C -- Invokes --> D
        C -- Invokes --> F
        C -- Invokes --> G
        C -- Invokes --> H
        D -- Delegates to --> E
    end
    
    subgraph "Signals / Slots"
        C -- preview_updated(ImageData) --> PreviewWidget[PreviewWidget updates display]
        C -- params_changed(Params) --> B[ParameterPanel updates sliders]
        C -- status_message_changed(str) --> A[MainWindow updates status bar]
        B -- parameter_changed() --> A[MainWindow calls context.update_params()]
    end

    style MainWindow fill:#cde4ff,stroke:#333,stroke-width:2px
    style ParameterPanel fill:#cde4ff,stroke:#333,stroke-width:2px
    style PreviewWidget fill:#cde4ff,stroke:#333,stroke-width:2px
    style ApplicationContext fill:#ffd4b3,stroke:#333,stroke-width:4px
    style TheEnlarger fill:#d4ffb3,stroke:#333,stroke-width:2px
    style FilmPipelineProcessor fill:#d4ffb3,stroke:#333,stroke-width:2px
    style ImageManager fill:#d4ffb3,stroke:#333,stroke-width:2px
    style ColorSpaceManager fill:#d4ffb3,stroke:#333,stroke-width:2px
    style PresetManager fill:#d4ffb3,stroke:#333,stroke-width:2px

```

## 模块职责划分

### UI Layer (UI 层)
*   `MainWindow`:
    *   **职责**: 应用主入口，窗口和菜单的容器。
    *   **功能**: 创建并组织其他UI组件（如 `ParameterPanel`, `PreviewWidget`）和 `ApplicationContext`。它负责将用户的顶层操作（如点击“打开文件”菜单）转发给 `ApplicationContext`，并连接 `Context` 的信号来更新非特定业务的UI部分（如状态栏）。
*   `ParameterPanel`:
    *   **职责**: 显示和编辑所有调色参数。
    *   **功能**: 完全由数据驱动。它订阅 `Context` 的 `params_changed` 信号，一旦收到新的 `ColorGradingParams` 对象，就用其数据（包括数值和名称）**完整更新**所有UI控件。当用户调整UI时，它会发出 `parameter_changed` 信号，通知 `MainWindow` 从它这里获取最新的、包含完整UI状态的 `ColorGradingParams` 对象。
*   `PreviewWidget`:
    *   **职责**: 显示处理后的预览图像。
    *   **功能**: 订阅 `ApplicationContext` 的 `preview_updated` 信号。一旦收到新的 `ImageData`，就将其渲染到屏幕上。

### Core Layer (核心层)
*   `ApplicationContext` (应用上下文/大脑):
    *   **职责**: **应用的单一数据源 (Single Source of Truth)**。管理所有核心状态和业务逻辑。
    *   **功能**:
        *   持有核心状态：`current_image`, `current_proxy` 以及**包含完整UI状态**的 `current_params`。
        *   持有核心服务实例：`ImageManager`, `TheEnlarger` 等。
        *   包含核心业务流程：`load_image()`, `update_params()`, `reset_params()` 等。
        *   通过后台线程池管理图像处理任务，避免UI阻塞。
        *   通过**统一的 `params_changed` 信号**将完整的状态变更通知给UI层。
*   `TheEnlarger`:
    *   **职责**: 图像处理管线的高级接口 (Facade)。
    *   **功能**: 提供简洁的API（如 `apply_full_pipeline`, `apply_preview_pipeline`），并将具体的处理任务委托给 `FilmPipelineProcessor`。
*   `FilmPipelineProcessor`:
    *   **职责**: 实际执行图像处理管线。
    *   **功能**: 实现从图像加载到最终输出的所有数学步骤，包括密度反相、矩阵校正、RGB增益、曲线应用等。它现在也负责加载和管理校正矩阵。
*   `ImageManager`, `ColorSpaceManager`, `PresetManager`:
    *   **职责**: 提供具体的、无状态的服务。
    *   **功能**: 分别负责图像的读写、色彩空间转换和预设文件的管理。

## 信号流 (Signals/Slots)

图中清晰地展示了关键的信号流：
1.  **用户操作**: 用户在 `ParameterPanel` 上改变一个参数，`ParameterPanel` 发出 `parameter_changed` 信号。
2.  **UI 协调**: `MainWindow` 监听到该信号，调用 `parameter_panel.get_current_params()` 获取包含**完整UI状态**（数值+名称）的新参数对象，然后调用 `context.update_params()` 将其传递给Context。
3.  **Context 处理**: `ApplicationContext` 更新自己的内部状态 (`_current_params`)，然后触发后台线程进行预览计算。
4.  **状态更新通知**: `ApplicationContext` 在其状态（`_current_params`）因任何原因（用户操作、加载预设、重置）改变后，发出：
    *   `params_changed(ColorGradingParams)` 信号。`ParameterPanel` 接收这个包含**完整UI状态**的对象，并用它来同步所有控件，包括滑块和下拉菜单（如果名称在标准列表中找不到，则添加带 `*` 的临时项）。
    *   `preview_updated(ImageData)` 信号（在计算完成后），`PreviewWidget` 接收后更新图像。
    *   `status_message_changed(str)` 信号，`MainWindow` 接收后更新状态栏文本。

这套架构有效地将“做什么”（UI层）和“怎么做”（核心层）分离开来，使得代码更加清晰、模块化，并易于未来的扩展。

## 光谱锐化（硬件校正）（Spectral Sharpening）

### 目标
- 基于 ColorChecker 24 色块，从实际扫描图像中优化扫描仪输入色彩变换（primaries），并联合微调密度域参数（gamma、dmax）与 RB 对数增益。
- 保持核心数学与管线不变，仅通过接口层完成编排与落地。

### 模块与职责
- UI 层：
  - `ParameterPanel`：
    - 提供“扫描仪光谱锐化（硬件校正）”入口（开关、色卡选择器、优化/保存按钮）。
    - 发出 `ccm_optimize_requested`、`save_custom_colorspace_requested`、`toggle_color_checker_requested` 等信号。
  - `PreviewWidget`：
    - 色卡选择器交互（四角点），负责在预览中叠加网格与参考色块。
  - `MainWindow`：
    - 作为一次性动作协调者：监听 `ccm_optimize_requested`，后台执行优化，应用结果（注册自定义输入空间、更新参数），并通过 `ApplicationContext` 触发预览刷新；监听 `save_custom_colorspace_requested` 写入用户配置。
- 核心/工具层：
  - `divere/utils/spectral_sharpening.py`：
    - 轻量胶水模块：线性化图像（按输入空间 gamma）、调用色卡提取、调用优化器，返回优化参数；不改动算法实现。
  - `divere/utils/ccm_optimizer/*`：
    - 已有优化管线与提取器：`optimizer.py`（CMA-ES）、`extractor.py`（色块提取）、`pipeline.py`（管线模拟器）。
  - `ColorSpaceManager`：
    - `register_custom_colorspace()` 动态注册 `<base>_custom` 输入空间；`get_color_space_info()` 提供输入空间 gamma 信息。
  - `ApplicationContext`：
    - `set_input_color_space()` 应用输入空间并重建代理；`update_params()` 合并参数并触发 `params_changed` 与后台预览；`preview_updated` 驱动 UI 刷新。

### 数据流
1. 用户在 `ParameterPanel` 勾选色卡选择器并调整四角点 → 点击“根据色卡计算光谱锐化（硬件校正）转换”。
2. `ParameterPanel` 发出 `ccm_optimize_requested` → `MainWindow` 响应：
   - 读取当前图像数组与输入空间 gamma、色卡角点、密度校正矩阵启用状态；
   - 后台调用 `divere/utils/spectral_sharpening.run()`：
     - 线性化图像（逆 gamma）→ 色卡提取（24 块线性 RGB）→ 调用 `CCMOptimizer` 执行 CMA-ES；
     - 返回 `primaries_xy`、`gamma`、`dmax`、`r_gain`、`b_gain` 与 RMSE 评估。
3. `MainWindow` 接收结果：
   - 使用 `ColorSpaceManager.register_custom_colorspace(<base>_custom, primaries_xy)` 注册自定义输入空间；
   - 调用 `ApplicationContext.set_input_color_space(<base>_custom)` 切换输入空间（内部重建代理）；
   - 获取当前 `ColorGradingParams`，合并 `gamma/dmax` 与 RB 对数增益，`context.update_params(new_params)`；
   - `ApplicationContext` 触发预览后台任务 → `preview_updated(ImageData)` → `PreviewWidget` 刷新。
4. 用户可点击“保存输入色彩变换结果”，`MainWindow` 将 UCS 三角坐标保存为用户目录 `config/colorspace/<name>_custom.json`（UTF-8，`ensure_ascii=False`）。

### 光谱锐化（硬件校正）交互图 (Mermaid.js)

```mermaid
graph TD
  subgraph UI
    PP[ParameterPanel]
    PV[PreviewWidget]
    MW[MainWindow]
  end

  subgraph Core/Utils
    CTX[ApplicationContext]
    CSM[ColorSpaceManager]
    SS[utils/spectral_sharpening.run]
    OPT[ccm_optimizer (CMA-ES)]
    EXT[extractor (ColorChecker)]
  end

  PP -- ccm_optimize_requested --> MW
  MW -- get corners/gamma/matrix --> PV
  MW -- invoke --> SS
  SS -- uses --> EXT
  SS -- uses --> OPT
  SS -- returns params --> MW
  MW -- register_custom_colorspace --> CSM
  MW -- set_input_color_space --> CTX
  MW -- update_params --> CTX
  CTX -- params_changed --> PP
  CTX -- preview_updated --> PV
```

### 设计约束与非目标
- 不修改 `math_ops`、`pipeline_processor`、`the_enlarger` 的核心算法与流程；光谱锐化（硬件校正）仅作为接口层功能接入。
- 色卡提取与优化调用在后台线程运行，避免阻塞 UI；预览线程由 `ApplicationContext` 统一管理。
- 自定义输入色彩空间仅注册于运行时；是否持久化由用户显式保存控制。
