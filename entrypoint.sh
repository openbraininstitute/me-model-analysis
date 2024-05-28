#!/bin/sh
timeout 3600 uvicorn me_model_analysis.main:APP --host 0.0.0.0 --port 8080 --proxy-headers --log-config logging.yaml
