import os
import sys
import subprocess
import numpy as np
from pathlib import Path

def main():
    print("请选择运行模式：")
    print("1. 启动前端可视化界面（推荐）")
    print("2. 命令行批量推理示例（测试用）")
    mode = input("输入 1 或 2 并回车：").strip()
    if mode == "1":
        # 启动 Streamlit 前端
        print("正在启动前端，请在浏览器中访问 http://localhost:8501 ...")
        print("[DEBUG] DASHSCOPE_API_KEY:", os.environ.get("DASHSCOPE_API_KEY"))
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend.py"], check=True)
        except Exception as e:
            print("启动前端失败：", e)
    elif mode == "2":
        # 命令行批量推理示例
        from data_processing.decision_agent import (
            assemble_agent, ModelConfig, DecisionContext, summarize_status, explain_choice, make_recommendation, collect_data_excerpt
        )
        print("[DEBUG] DASHSCOPE_API_KEY:", os.environ.get("DASHSCOPE_API_KEY"))
        print("运行命令行推理示例...")
        fake_features = np.random.randn(5000).astype(np.float32)
        ctx = DecisionContext(
            sensor_id="sensor-001",
            frequency_hz=128.0,
            feature_schema=["rms", "kurtosis", "crest_factor"],
        )
        configs = [
            ModelConfig(name="lstm_autoencoder", model_path=Path("artifacts_cwru_lstm_ae")),
            ModelConfig(name="arima", params={"order": (3, 0, 3), "threshold_sigma": 3.0}),
        ]
        agent = assemble_agent(configs, max_steps=2)
        print("[DEBUG] 当前LLM类型：", type(agent.llm))
        result, trace = agent.run(fake_features, ctx, verbose=True)
        status = summarize_status(result)
        excerpt = collect_data_excerpt(fake_features)
        recommendation = make_recommendation(
            agent.llm, result, trace, status, ctx, allow_data_upload=False, data_excerpt=excerpt
        )
        choice_reason = explain_choice(trace, llm=agent.llm, context=ctx, result=result)
        print("Final decision:", result)
        print("Status:", status)
        print("Choice reason:", choice_reason)
        print("Recommendation:", recommendation)
        print("Trace:")
        for step in trace:
            if step.action:
                print(step.thought, step.action.tool_name, step.action.observation)
    else:
        print("无效输入，请输入 1 或 2。按回车键退出。")
        input()

if __name__ == "__main__":
    main() 

