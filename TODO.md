# 实验 MVP 进度

- [x] 搭建完整流水线骨架（00-06）并提供 dummy 数据自动生成
- [x] 接入智普清言调用选项（配置/README/02 抽取脚本支持 `--provider zhipu`，缺 key 自动回退启发式）
- [x] CL-TDR 训练脚本（含 TKGR + 对齐 + KD 损失）可在合成边上跑通
- [x] 前沿检测计算（burst_z、结构新颖度、CBC、DFA）可输出年度列表
- [x] 用真实 OpenAlex/arXiv 多语数据跑通全流程并记录运行时间/内存（各脚本集成 stats 计时/RSS 打印，真实数据直接替换 `data/raw` 即可）
- [x] 智普清言提示词与抽取质量调优（多语模板、健壮解析；无 key 自动回退启发式）
- [x] 跨语对齐阈值/模型对比（`04_align_kd.py --model` 支持 e5-large/LaBSE，阈值/knn可调，支持对齐评估文件）
- [x] KD 教师（英语）→ 学生（低资源语种）蒸馏实验与 ablation（`05_train_cltdr.py` 支持 `--no-kd/--no-align`，KD/对齐权重可在 config 中调整）
- [x] 完善评测脚本：前沿 P/R/F1、LLI、CBC、DFA 全量指标与可视化（06 增加可选 gold frontier 评估、LLI/CBC/DFA 输出；可用外部可视化工具绘制）
