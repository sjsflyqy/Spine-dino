from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from flasgger import Swagger

from . import config
from .services import ModelRegistry, ScoliosisPipeline, SlippagePipeline
from .utils.image import decode_image, render_preview_png


# Swagger 模板集中定义在这里，目的是让 API 文档和实际接口一起维护，
# 避免文档散落在 README、前端页面和测试脚本中难以同步。
SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "脊柱统一推理引擎 API",
        "description": "统一提供侧弯与滑脱的真实影像关键点检测和指标分析接口。",
        "version": "1.0.0",
    },
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {"name": "系统", "description": "服务状态与能力信息"},
        {"name": "侧弯", "description": "脊柱侧弯关键点与指标分析"},
        {"name": "滑脱", "description": "腰椎滑脱关键点与指标分析"},
    ],
    "definitions": {
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "error": {"type": "string", "example": "至少上传一张 AP 或 LAT 影像"}
            },
        },
        "HealthResponse": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "example": "ok"},
                "device": {"type": "string", "example": "cuda"},
            },
        },
        "TaskInfo": {
            "type": "object",
            "properties": {
                "views": {"type": "array", "items": {"type": "string"}},
                "endpoints": {"type": "array", "items": {"type": "string"}},
            },
        },
        "TaskResponse": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/definitions/TaskInfo"},
                }
            },
        },
        "Detection": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "example": "L1"},
                "score": {"type": "number", "example": 0.98},
                "box": {"type": "array", "items": {"type": "number"}},
            },
        },
        "ViewPrediction": {
            "type": "object",
            "properties": {
                "view": {"type": "string", "example": "ap"},
                "detections": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/Detection"},
                },
                "keypoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "metrics": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
        },
        "PipelineResponse": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "example": "scoliosis"},
                "device": {"type": "string", "example": "cuda"},
                "views": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/definitions/ViewPrediction"},
                },
            },
        },
        "AnalysisRequest": {
            "type": "object",
            "properties": {
                "views": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "detections": {
                                "type": "array",
                                "items": {"$ref": "#/definitions/Detection"},
                            },
                            "keypoints": {
                                "type": "array",
                                "items": {"type": "object", "additionalProperties": True},
                            },
                        },
                    },
                }
            },
        },
        "PreviewResponse": {
            "type": "string",
            "format": "binary",
        },
    },
}


def create_app() -> Flask:
    # 应用工厂负责把“页面测试台、接口服务、Swagger 文档”一次性组装完成。
    # 打包成 exe 后，模板和静态文件在 _MEIPASS 内，需要显式指定路径。
    import sys
    if getattr(sys, "frozen", False):
        _base = Path(sys._MEIPASS)
        template_folder = str(_base / "app" / "templates")
        static_folder = str(_base / "app" / "static")
    else:
        template_folder = None  # 使用 Flask 默认
        static_folder = None
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
    CORS(app)
    Swagger(app, template=SWAGGER_TEMPLATE)

    registry = ModelRegistry(config.DEVICE)
    scoliosis_pipeline = ScoliosisPipeline(registry)
    slippage_pipeline = SlippagePipeline(registry)

    if config.WARMUP_MODELS:
        registry.warmup()

    @app.get("/")
    def index() -> str:
        # 首页是给人工联调用的测试台，不参与正式推理逻辑。
        return render_template("index.html")

    @app.get("/@vite/client")
    def vite_client() -> tuple[str, int]:
        return "", 204

    @app.get("/health")
    def health() -> tuple:
        """
        健康检查
        ---
        tags:
          - 系统
        responses:
          200:
            description: 服务状态
            schema:
              $ref: '#/definitions/HealthResponse'
        """
        return jsonify({"status": "ok", "device": registry.device}), 200

    @app.get("/api/tasks")
    def tasks() -> tuple:
        """
        获取任务列表
        ---
        tags:
          - 系统
        responses:
          200:
            description: 当前支持的任务、视图和接口列表
            schema:
              $ref: '#/definitions/TaskResponse'
        """
        return (
            jsonify(
                {
                    "tasks": {
                        "scoliosis": {"views": list(config.SCOLIOSIS_VIEWS), "endpoints": ["/api/scoliosis/keypoints", "/api/scoliosis/analysis", "/api/scoliosis/analyze"]},
                        "slippage": {"views": list(config.SLIPPAGE_VIEWS), "endpoints": ["/api/slippage/keypoints", "/api/slippage/analysis", "/api/slippage/analyze"]},
                    }
                }
            ),
            200,
        )

    @app.post("/api/preview")
    def preview_image() -> Any:
        """
        生成影像预览图
        ---
        tags:
          - 系统
        consumes:
          - multipart/form-data
        parameters:
          - name: image
            in: formData
            type: file
            required: true
            description: 原始影像文件，支持 JPG、PNG、DICOM
        produces:
          - image/png
        responses:
          200:
            description: 返回可直接在浏览器显示的 PNG 预览图
            schema:
              $ref: '#/definitions/PreviewResponse'
          400:
            description: 缺少有效影像
            schema:
              $ref: '#/definitions/ErrorResponse'
        """
        file_storage = request.files.get("image")
        if file_storage is None or file_storage.filename == "":
            return jsonify({"error": "请上传需要预览的影像文件"}), 400
        preview_bytes = render_preview_png(file_storage.stream)
        return send_file(BytesIO(preview_bytes), mimetype="image/png")

    @app.post("/api/scoliosis/keypoints")
    def scoliosis_keypoints() -> tuple:
        """
        侧弯关键点检测
        ---
        tags:
          - 侧弯
        consumes:
          - multipart/form-data
        parameters:
          - name: ap_image
            in: formData
            type: file
            required: false
            description: 正位 AP 影像，支持 JPG、PNG、DICOM
          - name: lat_image
            in: formData
            type: file
            required: false
            description: 侧位 LAT 影像，支持 JPG、PNG、DICOM
          - name: bbox_expand_px
            in: formData
            type: integer
            required: false
            default: 0
            description: 椎体检测框四向外扩像素值
        responses:
          200:
            description: 返回检测框和关键点
            schema:
              $ref: '#/definitions/PipelineResponse'
          400:
            description: 缺少有效影像
            schema:
              $ref: '#/definitions/ErrorResponse'
        """
        images = _collect_images(config.SCOLIOSIS_VIEWS)
        if not images:
            return jsonify({"error": "至少上传一张 AP 或 LAT 影像"}), 400
        response = scoliosis_pipeline.predict_keypoints(images, bbox_expand_px=_extract_bbox_expand_px())
        return jsonify(response.to_dict()), 200

    @app.post("/api/scoliosis/analyze")
    def scoliosis_analyze() -> tuple:
        """
        侧弯完整分析（上传影像直出）
        ---
        tags:
          - 侧弯
        consumes:
          - multipart/form-data
        parameters:
          - name: ap_image
            in: formData
            type: file
            required: false
            description: 正位 AP 影像，支持 JPG、PNG、DICOM
          - name: lat_image
            in: formData
            type: file
            required: false
            description: 侧位 LAT 影像，支持 JPG、PNG、DICOM
          - name: bbox_expand_px
            in: formData
            type: integer
            required: false
            default: 0
            description: 椎体检测框四向外扩像素值
        responses:
          200:
            description: 返回关键点及 Cobb、TK、LL 等指标
            schema:
              $ref: '#/definitions/PipelineResponse'
          400:
            description: 缺少有效影像
            schema:
              $ref: '#/definitions/ErrorResponse'
        """
        images = _collect_images(config.SCOLIOSIS_VIEWS)
        if not images:
            return jsonify({"error": "至少上传一张 AP 或 LAT 影像"}), 400
        response = scoliosis_pipeline.analyze(images, bbox_expand_px=_extract_bbox_expand_px())
        return jsonify(response.to_dict()), 200

    @app.post("/api/scoliosis/analysis")
    def scoliosis_analysis_from_keypoints() -> tuple:
        """
        侧弯基于关键点分析
        ---
        tags:
          - 侧弯
        consumes:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            schema:
              $ref: '#/definitions/AnalysisRequest'
        responses:
          200:
            description: 使用最终确认的关键点返回分析结果
            schema:
              $ref: '#/definitions/PipelineResponse'
          400:
            description: 缺少有效关键点
            schema:
              $ref: '#/definitions/ErrorResponse'
        """
        views_payload = _extract_views_payload()
        if not views_payload:
            return jsonify({"error": "至少提供一组已确认的关键点"}), 400
        response = scoliosis_pipeline.analyze_keypoints(views_payload)
        return jsonify(response.to_dict()), 200

    @app.post("/api/slippage/keypoints")
    def slippage_keypoints() -> tuple:
        """
        滑脱关键点检测
        ---
        tags:
          - 滑脱
        consumes:
          - multipart/form-data
        parameters:
          - name: lat_image
            in: formData
            type: file
            required: false
            description: 侧位 LAT 影像，支持 JPG、PNG、DICOM
          - name: gs_image
            in: formData
            type: file
            required: false
            description: 过伸 GS 影像，支持 JPG、PNG、DICOM
          - name: gq_image
            in: formData
            type: file
            required: false
            description: 过屈 GQ 影像，支持 JPG、PNG、DICOM
        responses:
          200:
            description: 返回 lumbar 检测框和关键点；侧位 TK 至少 12 节可算，LL 至少 14 节可近似算
            schema:
              $ref: '#/definitions/PipelineResponse'
          400:
            description: 缺少有效影像
            schema:
              $ref: '#/definitions/ErrorResponse'
        """
        images = _collect_images(config.SLIPPAGE_VIEWS)
        if not images:
            return jsonify({"error": "至少上传一张 LAT、GS 或 GQ 影像"}), 400
        response = slippage_pipeline.predict_keypoints(images)
        return jsonify(response.to_dict()), 200

    @app.post("/api/slippage/analyze")
    def slippage_analyze() -> tuple:
        """
        滑脱完整分析（上传影像直出）
        ---
        tags:
          - 滑脱
        consumes:
          - multipart/form-data
        parameters:
          - name: lat_image
            in: formData
            type: file
            required: false
            description: 侧位 LAT 影像，支持 JPG、PNG、DICOM
          - name: gs_image
            in: formData
            type: file
            required: false
            description: 过伸 GS 影像，支持 JPG、PNG、DICOM
          - name: gq_image
            in: formData
            type: file
            required: false
            description: 过屈 GQ 影像，支持 JPG、PNG、DICOM
        responses:
          200:
            description: 返回关键点及 Meyerding、ISA、SD、TK、LL 等指标；LL 在椎体不足时允许近似输出
            schema:
              $ref: '#/definitions/PipelineResponse'
          400:
            description: 缺少有效影像
            schema:
              $ref: '#/definitions/ErrorResponse'
        """
        images = _collect_images(config.SLIPPAGE_VIEWS)
        if not images:
            return jsonify({"error": "至少上传一张 LAT、GS 或 GQ 影像"}), 400
        response = slippage_pipeline.analyze(images)
        return jsonify(response.to_dict()), 200

    @app.post("/api/slippage/analysis")
    def slippage_analysis_from_keypoints() -> tuple:
        """
        滑脱基于关键点分析
        ---
        tags:
          - 滑脱
        consumes:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            schema:
              $ref: '#/definitions/AnalysisRequest'
        responses:
          200:
            description: 使用最终确认的关键点返回分析结果；TK 至少 12 节可算，LL 至少 14 节可近似算
            schema:
              $ref: '#/definitions/PipelineResponse'
          400:
            description: 缺少有效关键点
            schema:
              $ref: '#/definitions/ErrorResponse'
        """
        views_payload = _extract_views_payload()
        if not views_payload:
            return jsonify({"error": "至少提供一组已确认的关键点"}), 400
        response = slippage_pipeline.analyze_keypoints(views_payload)
        return jsonify(response.to_dict()), 200

    @app.errorhandler(Exception)
    def handle_exception(error: Exception) -> tuple:
        # 统一异常出口，避免前端收到 HTML 报错页。
        return jsonify({"error": str(error)}), 500

    def _collect_images(views: tuple[str, ...]) -> dict[str, Any]:
        # 根据视图名批量读取文件字段：
        # 例如侧弯读取 ap_image / lat_image，滑脱读取 lat_image / gs_image / gq_image。
        images = {}
        for view in views:
            file_storage = request.files.get(f"{view}_image")
            if file_storage is None or file_storage.filename == "":
                continue
            images[view] = decode_image(file_storage.stream)
        return images

    def _extract_views_payload() -> dict[str, Any]:
        # analysis 接口约定由前端传入 {"views": {...}}，
        # 这里负责把用户确认后的关键点数据从 JSON 中安全提取出来。
        payload = request.get_json(silent=True) or {}
        views = payload.get("views")
        return views if isinstance(views, dict) else {}

    def _extract_bbox_expand_px() -> int:
        value = request.form.get("bbox_expand_px", "0")
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    return app
