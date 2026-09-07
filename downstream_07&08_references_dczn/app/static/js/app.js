// 这个配置对象把“任务名、视图、接口、按钮、点位命中半径”统一收口。
// 后续无论是新增任务还是改接口路径，都尽量只改这里一处。
const TASKS = {
    scoliosis: {
        views: ["ap", "lat"],
        keypointsEndpoint: "/api/scoliosis/keypoints",
        analyzeEndpoint: "/api/scoliosis/analysis",
        outputId: "scoliosis-output",
        logId: "scoliosis-log",
        bboxInputId: "scoliosis-bbox-expand-px",
        buttons: {
            keypoints: "scoliosis-keypoints-btn",
            analyze: "scoliosis-analyze-btn",
            reset: "scoliosis-reset-btn"
        },
        pointRadius: 8
    },
    slippage: {
        views: ["lat", "gs", "gq"],
        keypointsEndpoint: "/api/slippage/keypoints",
        analyzeEndpoint: "/api/slippage/analysis",
        outputId: "slippage-output",
        logId: "slippage-log",
        buttons: {
            keypoints: "slippage-keypoints-btn",
            analyze: "slippage-analyze-btn",
            reset: "slippage-reset-btn"
        },
        pointRadius: 9
    }
};

// 前端状态拆成四类：
// files 保存原始上传文件，previews 保存可显示图片，
// originalViews 保存模型原始关键点，workingViews 保存用户编辑后的关键点。
const state = {
    scoliosis: { files: {}, previews: {}, originalViews: {}, workingViews: {}, drag: null, zoom: {} },
    slippage: { files: {}, previews: {}, originalViews: {}, workingViews: {}, drag: null, zoom: {} }
};

const $ = (id) => document.getElementById(id);

function setLoading(visible, text = "请求处理中...") {
    $("loading-mask").classList.toggle("hidden", !visible);
    $("loading-text").textContent = text;
}

function setLog(task, text) {
    $(TASKS[task].logId).textContent = text;
}

function setOutput(task, payload) {
    $(TASKS[task].outputId).textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
}

function setButtonsDisabled(task, disabled) {
    const ids = TASKS[task].buttons;
    $(ids.keypoints).disabled = disabled;
    $(ids.analyze).disabled = disabled;
    $(ids.reset).disabled = disabled;
}

function getBBoxExpandPx(task) {
    const inputId = TASKS[task].bboxInputId;
    if (!inputId) {
        return 0;
    }
    const input = $(inputId);
    if (!input) {
        return 0;
    }
    const value = Number.parseInt(input.value || "0", 10);
    return Number.isFinite(value) && value > 0 ? value : 0;
}

function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach((item) => item.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
            button.classList.add("active");
            $(`tab-${button.dataset.tab}`).classList.add("active");
        });
    });
}

async function checkHealth() {
    try {
        const response = await fetch("/health");
        const data = await response.json();
        $("health-badge").textContent = "服务正常";
        $("health-badge").className = "badge badge-ok";
        $("device-badge").textContent = `设备 ${data.device}`;
    } catch (error) {
        $("health-badge").textContent = "服务异常";
        $("health-badge").className = "badge badge-error";
        $("device-badge").textContent = "设备未知";
    }
}

function initTask(task) {
    TASKS[task].views.forEach((view) => {
        const input = $(`${task}-${view}-file`);
        const fileName = $(`${task}-${view}-name`);
        const canvas = $(`${task}-${view}-canvas`);
        if (!input || !fileName || !canvas) {
            return;
        }
        state[task].zoom[view] = 1;
        input.addEventListener("change", async (event) => {
            const file = event.target.files[0];
            state[task].files[view] = file || null;
            fileName.textContent = file ? file.name : "未选择文件";
            if (!file) {
                state[task].previews[view] = null;
                state[task].originalViews[view] = null;
                state[task].workingViews[view] = null;
                clearCanvas(canvas);
                resetViewport(task, view);
                return;
            }
            try {
                const preview = await loadRenderableImage(file);
                state[task].previews[view] = preview;
                if (preview) {
                    drawBaseImage(canvas, preview);
                    fitViewport(task, view);
                } else {
                    clearCanvas(canvas, "该文件可上传推理，但浏览器当前无法预览");
                    resetViewport(task, view);
                }
            } catch (error) {
                state[task].previews[view] = null;
                clearCanvas(canvas, "预览失败，但仍可继续上传测试");
                resetViewport(task, view);
            }
            state[task].originalViews[view] = null;
            state[task].workingViews[view] = null;
            setOutput(task, "暂无结果");
            setLog(task, "已更新影像，请先执行关键点检测。");
        });
        bindCanvasEditing(task, view, canvas);
        bindViewportControls(task, view);
    });

    $(TASKS[task].buttons.keypoints).addEventListener("click", () => requestKeypoints(task));
    $(TASKS[task].buttons.analyze).addEventListener("click", () => analyzeCurrentKeypoints(task));
    $(TASKS[task].buttons.reset).addEventListener("click", () => resetKeypoints(task));
}

async function requestKeypoints(task) {
    // 第一步：上传影像，拿到模型初始关键点。
    const files = state[task].files;
    const selectedViews = TASKS[task].views.filter((view) => files[view]);
    if (selectedViews.length === 0) {
        setLog(task, "请先选择至少一张影像。");
        return;
    }

    const formData = new FormData();
    const bboxExpandPx = getBBoxExpandPx(task);
    selectedViews.forEach((view) => formData.append(`${view}_image`, files[view]));
    if (task === "scoliosis") {
        formData.append("bbox_expand_px", String(bboxExpandPx));
    }

    setButtonsDisabled(task, true);
    setLoading(true, `${task === "scoliosis" ? "侧弯" : "滑脱"}关键点检测中...`);
    const logLines = [`第 1 步：上传影像并检测关键点`, `视图: ${selectedViews.join(", ")}`];
    if (task === "scoliosis") {
        logLines.push(`框外扩: ${bboxExpandPx} px`);
    }
    setLog(task, logLines.join("\n"));

    try {
        const startedAt = performance.now();
        const response = await fetch(TASKS[task].keypointsEndpoint, {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        const elapsed = ((performance.now() - startedAt) / 1000).toFixed(2);
        if (!response.ok) {
            throw new Error(data.error || "请求失败");
        }
        state[task].originalViews = deepClone(data.views || {});
        state[task].workingViews = deepClone(data.views || {});
        renderTask(task);
        setOutput(task, data);
        setLog(task, `第 1 步完成：关键点已生成\n可直接拖拽画布上的点位进行微调\n耗时: ${elapsed}s\n设备: ${data.device}`);
    } catch (error) {
        setLog(task, `请求失败\n${error.message}`);
        setOutput(task, { error: error.message });
    } finally {
        setLoading(false);
        setButtonsDisabled(task, false);
    }
}

async function analyzeCurrentKeypoints(task) {
    // 第二步：不再重新上传影像，而是把“当前画布上的最终关键点”发给后端分析。
    const views = state[task].workingViews;
    if (!views || Object.keys(views).length === 0) {
        setLog(task, "请先执行关键点检测，再分析当前关键点。");
        return;
    }
    const payload = { views: buildAnalysisPayload(task, views) };
    setButtonsDisabled(task, true);
    setLoading(true, `${task === "scoliosis" ? "侧弯" : "滑脱"}关键点分析中...`);
    setLog(task, "第 2 步：使用当前关键点进行分析");
    try {
        const startedAt = performance.now();
        const response = await fetch(TASKS[task].analyzeEndpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        const elapsed = ((performance.now() - startedAt) / 1000).toFixed(2);
        if (!response.ok) {
            throw new Error(data.error || "请求失败");
        }
        state[task].workingViews = mergeMetricsIntoViews(state[task].workingViews, data.views || {});
        renderTask(task);
        setOutput(task, data);
        setLog(task, `第 2 步完成：已使用最终关键点完成分析\n耗时: ${elapsed}s\n设备: ${data.device}`);
    } catch (error) {
        setLog(task, `分析失败\n${error.message}`);
        setOutput(task, { error: error.message });
    } finally {
        setLoading(false);
        setButtonsDisabled(task, false);
    }
}

function resetKeypoints(task) {
    // 用户拖拽过关键点之后，可以一键恢复到模型首次输出的原始点位。
    if (!state[task].originalViews || Object.keys(state[task].originalViews).length === 0) {
        setLog(task, "当前没有可重置的关键点。");
        return;
    }
    state[task].workingViews = deepClone(state[task].originalViews);
    renderTask(task);
    setLog(task, "已恢复到模型初始关键点。");
}

function renderTask(task) {
    const views = state[task].workingViews || {};
    TASKS[task].views.forEach((view) => {
        const canvas = $(`${task}-${view}-canvas`);
        const preview = state[task].previews[view];
        if (preview) {
            drawBaseImage(canvas, preview);
        } else {
            clearCanvas(canvas, "暂无预览");
        }
        const result = views[view];
        if (!result) {
            return;
        }
        if (preview) {
            drawKeypoints(task, canvas, result.keypoints || []);
            if (task === "scoliosis" && view === "ap") {
                drawCobbAngles(canvas, getScoliosisCobbAngles(result));
            }
        }
        syncViewport(task, view);
    });
}

function clearCanvas(canvas, message = "等待影像") {
    const width = canvas.width || 640;
    const height = canvas.height || 420;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0b1220";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "18px Microsoft YaHei";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(message, width / 2, height / 2);
}

function drawBaseImage(canvas, image) {
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, image.width, image.height);
}

function getViewportElements(task, view) {
    return {
        viewport: $(`${task}-${view}-viewport`),
        canvas: $(`${task}-${view}-canvas`),
        label: $(`${task}-${view}-zoom-label`)
    };
}

function getZoomBounds() {
    return { min: 0.25, max: 8 };
}

function getWidthFitZoom(task, view) {
    const { viewport, canvas } = getViewportElements(task, view);
    if (!viewport || !canvas || !canvas.width) {
        return 1;
    }
    const availableWidth = Math.max(120, viewport.clientWidth - 24);
    const zoom = availableWidth / canvas.width;
    return Math.min(getZoomBounds().max, Math.max(getZoomBounds().min, zoom));
}

function updateZoomLabel(task, view) {
    const { label } = getViewportElements(task, view);
    if (!label) {
        return;
    }
    const zoom = state[task].zoom[view] || 1;
    label.textContent = `${Math.round(zoom * 100)}%`;
}

function applyZoom(task, view, nextZoom, anchor = null) {
    const { viewport, canvas } = getViewportElements(task, view);
    if (!viewport || !canvas || !canvas.width || !canvas.height) {
        return;
    }
    const { min, max } = getZoomBounds();
    const zoom = Math.min(max, Math.max(min, nextZoom));
    const prevWidth = canvas.clientWidth || canvas.width;
    const prevHeight = canvas.clientHeight || canvas.height;
    const xRatio = anchor?.xRatio ?? ((viewport.scrollLeft + viewport.clientWidth / 2) / Math.max(prevWidth, 1));
    const yRatio = anchor?.yRatio ?? ((viewport.scrollTop + viewport.clientHeight / 2) / Math.max(prevHeight, 1));
    state[task].zoom[view] = zoom;
    canvas.style.width = `${Math.max(1, Math.round(canvas.width * zoom))}px`;
    canvas.style.height = `${Math.max(1, Math.round(canvas.height * zoom))}px`;
    const nextWidth = canvas.clientWidth || canvas.width;
    const nextHeight = canvas.clientHeight || canvas.height;
    viewport.scrollLeft = Math.max(0, xRatio * nextWidth - viewport.clientWidth / 2);
    viewport.scrollTop = Math.max(0, yRatio * nextHeight - viewport.clientHeight / 2);
    updateZoomLabel(task, view);
}

function fitViewport(task, view) {
    applyZoom(task, view, getWidthFitZoom(task, view), { xRatio: 0.5, yRatio: 0.5 });
}

function resetViewport(task, view) {
    state[task].zoom[view] = 1;
    const { viewport, canvas } = getViewportElements(task, view);
    if (canvas) {
        canvas.style.width = "";
        canvas.style.height = "";
    }
    if (viewport) {
        viewport.scrollLeft = 0;
        viewport.scrollTop = 0;
    }
    updateZoomLabel(task, view);
}

function syncViewport(task, view) {
    const canvas = $(`${task}-${view}-canvas`);
    if (!canvas || !canvas.width || !canvas.height) {
        return;
    }
    const currentZoom = state[task].zoom[view];
    if (!currentZoom || currentZoom === 1) {
        fitViewport(task, view);
        return;
    }
    applyZoom(task, view, currentZoom);
}

function bindViewportControls(task, view) {
    const { viewport, canvas } = getViewportElements(task, view);
    const zoomInButton = $(`${task}-${view}-zoom-in`);
    const zoomOutButton = $(`${task}-${view}-zoom-out`);
    const zoomResetButton = $(`${task}-${view}-zoom-reset`);
    if (!viewport || !canvas || !zoomInButton || !zoomOutButton || !zoomResetButton) {
        return;
    }
    zoomInButton.addEventListener("click", () => applyZoom(task, view, (state[task].zoom[view] || 1) * 1.25));
    zoomOutButton.addEventListener("click", () => applyZoom(task, view, (state[task].zoom[view] || 1) / 1.25));
    zoomResetButton.addEventListener("click", () => fitViewport(task, view));
    viewport.addEventListener("wheel", (event) => {
        if (!event.ctrlKey && !event.metaKey) {
            return;
        }
        event.preventDefault();
        const rect = viewport.getBoundingClientRect();
        const width = canvas.clientWidth || canvas.width || 1;
        const height = canvas.clientHeight || canvas.height || 1;
        const anchor = {
            xRatio: (viewport.scrollLeft + (event.clientX - rect.left)) / width,
            yRatio: (viewport.scrollTop + (event.clientY - rect.top)) / height
        };
        const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
        applyZoom(task, view, (state[task].zoom[view] || 1) * factor, anchor);
    }, { passive: false });
}

function drawKeypoints(task, canvas, groups) {
    const ctx = canvas.getContext("2d");
    groups.forEach((group, index) => {
        const color = `hsl(${(index * 29 + 120) % 360} 80% 58%)`;
        const points = getEditablePoints(task, group);
        ctx.fillStyle = color;
        points.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, task === "slippage" ? 2.5 : 2, 0, Math.PI * 2);
            ctx.fill();
        });
    });
}

function drawCobbAngles(canvas, cobbList) {
    if (!Array.isArray(cobbList) || cobbList.length === 0) {
        return;
    }
    const ctx = canvas.getContext("2d");
    cobbList.forEach((item, index) => {
        const endpoints = item?.endpoints;
        if (!endpoints) {
            return;
        }
        const startLeft = toPoint(endpoints.start_left);
        const startRight = toPoint(endpoints.start_right);
        const endLeft = toPoint(endpoints.end_left);
        const endRight = toPoint(endpoints.end_right);
        if (!startLeft || !startRight || !endLeft || !endRight) {
            return;
        }
        const color = `hsl(${(index * 53 + 8) % 360} 90% 62%)`;
        const startLine = extendSegment(startLeft, startRight, canvas.width * 0.18);
        const endLine = extendSegment(endLeft, endRight, canvas.width * 0.18);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(startLine.p1[0], startLine.p1[1]);
        ctx.lineTo(startLine.p2[0], startLine.p2[1]);
        ctx.moveTo(endLine.p1[0], endLine.p1[1]);
        ctx.lineTo(endLine.p2[0], endLine.p2[1]);
        ctx.stroke();

        const labelX = (startLeft[0] + startRight[0] + endLeft[0] + endRight[0]) / 4;
        const labelY = (startLeft[1] + startRight[1] + endLeft[1] + endRight[1]) / 4;
        const text = `${Number(item.angle || 0).toFixed(1)}°`;
        drawAngleLabel(ctx, text, labelX, labelY, color);
    });
}

function getScoliosisCobbAngles(viewResult) {
    const fromMetrics = viewResult?.metrics?.cobb;
    if (Array.isArray(fromMetrics) && fromMetrics.length > 0) {
        return fromMetrics;
    }
    return calculateCobbFromKeypoints(viewResult?.keypoints || []);
}

function calculateCobbFromKeypoints(groups) {
    if (!Array.isArray(groups) || groups.length < 2) {
        return [];
    }
    const pairs = [];
    for (let upperIndex = 0; upperIndex < groups.length; upperIndex += 1) {
        const upper = normalizeCobbGroup(groups[upperIndex]);
        if (!upper) {
            continue;
        }
        for (let lowerIndex = upperIndex + 1; lowerIndex < groups.length; lowerIndex += 1) {
            const lower = normalizeCobbGroup(groups[lowerIndex]);
            if (!lower) {
                continue;
            }
            const angle = angleBetweenLines(upper.topLeft, upper.topRight, lower.bottomLeft, lower.bottomRight);
            pairs.push({
                angle,
                indices: [upperIndex, lowerIndex],
                endpoints: {
                    start_left: upper.topLeft,
                    start_right: upper.topRight,
                    end_left: lower.bottomLeft,
                    end_right: lower.bottomRight
                }
            });
        }
    }
    pairs.sort((a, b) => b.angle - a.angle);
    const selected = [];
    const usedRanges = [];
    for (const item of pairs) {
        const [start, end] = item.indices;
        const overlaps = usedRanges.some(([usedStart, usedEnd]) => !(end < usedStart || start > usedEnd));
        if (overlaps) {
            continue;
        }
        usedRanges.push([start, end]);
        selected.push({
            ...item,
            angle: Number(item.angle.toFixed(2))
        });
        if (selected.length >= Math.min(3, groups.length - 1)) {
            break;
        }
    }
    return selected;
}

function normalizeCobbGroup(group) {
    if (!Array.isArray(group) || group.length < 4) {
        return null;
    }
    const points = group
        .filter((point) => Array.isArray(point) && point.length >= 2)
        .map((point) => [Number(point[0]), Number(point[1])]);
    if (points.length < 4) {
        return null;
    }
    return {
        topLeft: points[0],
        topRight: points[1],
        bottomLeft: points[2],
        bottomRight: points[3]
    };
}

function angleBetweenLines(upperLeft, upperRight, lowerLeft, lowerRight) {
    const upperAngle = edgeAngle(upperLeft, upperRight);
    const lowerAngle = edgeAngle(lowerLeft, lowerRight);
    const cobbAngle = Math.abs(upperAngle - lowerAngle) * 180 / Math.PI;
    return cobbAngle > 90 ? 180 - cobbAngle : cobbAngle;
}

function edgeAngle(left, right) {
    return Math.atan2(right[1] - left[1], right[0] - left[0]);
}

function toPoint(value) {
    if (!Array.isArray(value) || value.length < 2) {
        return null;
    }
    return [Number(value[0]), Number(value[1])];
}

function extendSegment(p1, p2, extraLength) {
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    const length = Math.hypot(dx, dy) || 1;
    const ux = dx / length;
    const uy = dy / length;
    return {
        p1: [p1[0] - ux * extraLength, p1[1] - uy * extraLength],
        p2: [p2[0] + ux * extraLength, p2[1] + uy * extraLength]
    };
}

function drawAngleLabel(ctx, text, x, y, color) {
    ctx.font = "bold 18px Microsoft YaHei";
    const textWidth = ctx.measureText(text).width;
    const paddingX = 8;
    const paddingY = 6;
    const boxWidth = textWidth + paddingX * 2;
    const boxHeight = 30;
    const left = Math.max(8, Math.min(x - boxWidth / 2, ctx.canvas.width - boxWidth - 8));
    const top = Math.max(8, Math.min(y - boxHeight / 2, ctx.canvas.height - boxHeight - 8));
    ctx.fillStyle = "rgba(2, 6, 23, 0.72)";
    ctx.fillRect(left, top, boxWidth, boxHeight);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(left, top, boxWidth, boxHeight);
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, left + paddingX, top + boxHeight / 2);
}

function getEditablePoints(task, group) {
    // 两个任务的关键点结构不同：
    // 侧弯本身就是 [[x, y], ...]，滑脱则是展平的一维数组，需要这里做适配。
    if (!Array.isArray(group)) {
        return [];
    }
    if (task === "scoliosis") {
        return group.filter((point) => Array.isArray(point) && point.length >= 2).map((point) => [point[0], point[1]]);
    }
    const values = group.map((value) => Number(value));
    if (values.length < 10) {
        return [];
    }
    return [
        [values[2], values[3]],
        [values[4], values[5]],
        [values[6], values[7]],
        [values[8], values[9]]
    ];
}

function setEditablePoint(task, group, pointIndex, x, y) {
    if (task === "scoliosis") {
        group[pointIndex] = [x, y];
        return;
    }
    const mapping = [
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]
    ];
    const indices = mapping[pointIndex];
    if (!indices) {
        return;
    }
    group[indices[0]] = x;
    group[indices[1]] = y;
    // 滑脱关键点里前两个位置保存的是中心点，拖拽角点后这里同步重算中心点。
    group[0] = (Number(group[2]) + Number(group[4]) + Number(group[6]) + Number(group[8])) / 4;
    group[1] = (Number(group[3]) + Number(group[5]) + Number(group[7]) + Number(group[9])) / 4;
}

function bindCanvasEditing(task, view, canvas) {
    // 这里实现的是最基础的拖点编辑：
    // 命中最近点 -> 按住拖动 -> 实时重绘。
    const startDrag = (event) => {
        const workingView = state[task].workingViews[view];
        if (!workingView || !workingView.keypoints) {
            return;
        }
        const position = getCanvasPoint(canvas, event);
        const hit = findClosestPoint(task, workingView.keypoints, position.x, position.y, TASKS[task].pointRadius * 1.8);
        if (!hit) {
            return;
        }
        state[task].drag = { view, groupIndex: hit.groupIndex, pointIndex: hit.pointIndex };
    };

    const moveDrag = (event) => {
        if (!state[task].drag || state[task].drag.view !== view) {
            return;
        }
        const workingView = state[task].workingViews[view];
        if (!workingView) {
            return;
        }
        const group = workingView.keypoints[state[task].drag.groupIndex];
        if (!group) {
            return;
        }
        const position = getCanvasPoint(canvas, event);
        setEditablePoint(task, group, state[task].drag.pointIndex, position.x, position.y);
        renderTask(task);
    };

    const endDrag = () => {
        state[task].drag = null;
    };

    canvas.addEventListener("mousedown", startDrag);
    canvas.addEventListener("mousemove", moveDrag);
    canvas.addEventListener("mouseup", endDrag);
    canvas.addEventListener("mouseleave", endDrag);
}

function getCanvasPoint(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY
    };
}

function findClosestPoint(task, groups, x, y, threshold) {
    // 找离鼠标最近的关键点，用于判断用户当前想拖动哪一个点。
    let best = null;
    let bestDistance = threshold;
    groups.forEach((group, groupIndex) => {
        getEditablePoints(task, group).forEach((point, pointIndex) => {
            const distance = Math.hypot(point[0] - x, point[1] - y);
            if (distance <= bestDistance) {
                bestDistance = distance;
                best = { groupIndex, pointIndex };
            }
        });
    });
    return best;
}

function buildAnalysisPayload(task, views) {
    // analysis 接口只需要最终确认的检测框和关键点，
    // 因此前端在这里把 workingViews 转成精简 payload。
    const payload = {};
    Object.entries(views).forEach(([view, data]) => {
        payload[view] = {
            detections: deepClone(data.detections || []),
            keypoints: task === "scoliosis"
                ? deepClone(data.keypoints || [])
                : (data.keypoints || []).map((group) => group.map((value) => Number(value)))
        };
    });
    return payload;
}

function mergeMetricsIntoViews(currentViews, responseViews) {
    // 分析完成后保留用户当前编辑过的关键点，只替换后端重新计算出来的 metrics。
    const merged = deepClone(currentViews);
    Object.entries(responseViews || {}).forEach(([view, payload]) => {
        if (!merged[view]) {
            merged[view] = payload;
            return;
        }
        merged[view].metrics = payload.metrics || {};
    });
    return merged;
}

function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
}

async function loadRenderableImage(file) {
    // 预览优先级：
    // 普通图片直接本地显示，DICOM 先尝试浏览器本地解析，失败再走后端兜底。
    const lowerName = file.name.toLowerCase();
    if (lowerName.endsWith(".dcm") || lowerName.endsWith(".dicom")) {
        try {
            const preview = await loadDicomPreview(file);
            if (preview) {
                return preview;
            }
        } catch (error) {
            console.warn("本地 DICOM 预览失败，改走后端预览兜底", error);
        }
        return loadServerPreview(file);
    }
    return loadStandardPreview(file);
}

function loadStandardPreview(file) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = URL.createObjectURL(file);
    });
}

async function loadDicomPreview(file) {
    // 这里是纯浏览器侧 DICOM 预览逻辑，优点是快，缺点是兼容性一般。
    // 所以一旦这里失败，外层会自动回退到后端 /api/preview。
    const buffer = await file.arrayBuffer();
    const byteArray = new Uint8Array(buffer);
    const dataSet = dicomParser.parseDicom(byteArray);
    const rows = dataSet.uint16("x00280010");
    const cols = dataSet.uint16("x00280011");
    const bitsAllocated = dataSet.uint16("x00280100");
    const pixelRepresentation = dataSet.uint16("x00280103") || 0;
    const photometric = (dataSet.string("x00280004") || "").toUpperCase();
    const element = dataSet.elements.x7fe00010;
    if (!rows || !cols || !element) {
        return null;
    }
    let pixels;
    if (bitsAllocated === 16) {
        pixels = pixelRepresentation === 0
            ? new Uint16Array(byteArray.buffer, element.dataOffset, rows * cols)
            : new Int16Array(byteArray.buffer, element.dataOffset, rows * cols);
    } else {
        pixels = new Uint8Array(byteArray.buffer, element.dataOffset, rows * cols);
    }
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < pixels.length; i += 1) {
        const value = pixels[i];
        if (value < min) min = value;
        if (value > max) max = value;
    }
    const scale = max > min ? 255 / (max - min) : 1;
    const canvas = document.createElement("canvas");
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(cols, rows);
    for (let i = 0; i < pixels.length; i += 1) {
        let value = Math.round((pixels[i] - min) * scale);
        if (photometric === "MONOCHROME1") {
            value = 255 - value;
        }
        imageData.data[i * 4] = value;
        imageData.data[i * 4 + 1] = value;
        imageData.data[i * 4 + 2] = value;
        imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    const image = new Image();
    return new Promise((resolve, reject) => {
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = canvas.toDataURL("image/png");
    });
}

async function loadServerPreview(file) {
    // 后端兜底预览可以处理更多压缩或特殊编码 DICOM，
    // 返回统一 PNG，这样浏览器一定能显示。
    const formData = new FormData();
    formData.append("image", file);
    const response = await fetch("/api/preview", {
        method: "POST",
        body: formData
    });
    if (!response.ok) {
        let message = "后端预览失败";
        try {
            const payload = await response.json();
            if (payload.error) {
                message = payload.error;
            }
        } catch (error) {
            console.warn("读取预览错误信息失败", error);
        }
        throw new Error(message);
    }
    const blob = await response.blob();
    return loadBlobPreview(blob);
}

function loadBlobPreview(blob) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        const url = URL.createObjectURL(blob);
        image.onload = () => {
            URL.revokeObjectURL(url);
            resolve(image);
        };
        image.onerror = (error) => {
            URL.revokeObjectURL(url);
            reject(error);
        };
        image.src = url;
    });
}

function init() {
    initTabs();
    checkHealth();
    initTask("scoliosis");
    initTask("slippage");
    Object.keys(TASKS).forEach((task) => {
        TASKS[task].views.forEach((view) => {
            clearCanvas($(`${task}-${view}-canvas`));
            resetViewport(task, view);
        });
    });
}

window.addEventListener("DOMContentLoaded", init);
