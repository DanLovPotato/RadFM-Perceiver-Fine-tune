# KE_generation：生成 Reg2RG 的 knowledge bank

## 这是干什么的

Reg2RG 在写某个病人某个器官的报告时，会参考 **5 份相似的 train 病人报告**。这个文件夹负责提前把这些准备好，最后生成两个文件给 Reg2RG 用：

| 文件 | 里面是什么 | 填到 Reg2RG 的 |
|---|---|---|
| `organ_report_embeddings.npz` | 每个 train 病人、每个器官的报告，各编码成一个 4096 维向量 | `bank_npy_path` |
| `organ_annotation.json` | 每个病人、每个器官，参考哪 5 个 train 病人的报告 | `organ_annotation_path` |

用到的模型都来自这个项目训练出的 checkpoint：文字侧用 `adapter`，图像侧用 `perceiver` + `fc`。

## 当前状态

两阶段检索已经实现（**不设阈值**），step145000 已经用全量数据跑完（train 24126 人，val 1564 人），结果见下面"零样本判断"和"两阶段的测试结果"。

如果想用旧的一阶段检索做对比，可以给 `build_organ_annotation.py` 加 `--mode plain`。

## 怎么跑

**1. 改设置。** 打开 `run_train.sh` 和 `run_test.sh`，改最上面几行。`ckpt_name`、`ckpt_dir`、`out_dir` 两个脚本必须写一样的；GPU 可以不同（默认 train 用 1 号、test 用 2 号）：

```bash
cuda_device="1"          # 用哪张 GPU
ckpt_name="step145000"   # 用哪个 step 的 checkpoint
ckpt_dir="..."           # checkpoint 所在的训练输出目录
out_dir="..."            # 结果存到哪里
```

数据（CT、mask、报告 csv）的路径在 [config.py](config.py) 里，见下面"config.py 是什么"。

**2. 先跑完 train，再跑 test（必须按顺序）：**

```bash
cd KE_generation
screen -dmS ke_train bash -c 'bash run_train.sh 2>&1 | tee ke_train.log'
# 等 run_train.sh 完全跑完，再启动 test
screen -dmS ke_test bash -c 'bash run_test.sh 2>&1 | tee ke_test.log'
```

test 要用 train 第 [3] 步算出的图像均值 (M)，所以必须等 train **全部**结束。

**3. 看结果。** 跑完最后会打印出要填进 Reg2RG 配置的路径。

| 脚本 | 大约多久 |
|---|---|
| `run_train.sh` | 约 5 小时（几乎都花在 [3] 编码 24126 个病人的 CT 上；[0]–[2] 只要几分钟） |
| `run_test.sh` | 约 30 分钟 |

## 跑完得到什么

都在 `/mnt/researchdrive/.../dataset/KE_bank_4096_<ckpt_name>/` 下：

```
KE_bank_4096_step145000/
├── organ_report_embeddings.npz                 ★ bank（train 和 val 都用这一个）
├── train/organ_annotation.json                 ★ train 用
├── val/organ_annotation.json                   ★ val 用
│
├── unique_reports.json                         (U) 中间文件
├── bank_labels.json                            (L) 中间文件：每句话正常/异常（新）
├── organ_report_embeddings_unique_4096.npz     (A) 中间文件
├── prototypes.npz                              (P) 中间文件：正常/异常原型（新）
├── train/image_mean.npz                        (M) 中间文件：train 图像均值，异常池去均值用（新）
├── train/decisions.npz、val/decisions.npz      每个病人每个器官的零样本判断，检查用（新）
├── train/image_queries.pt                      缓存
└── val/image_queries.pt                        缓存
```

Reg2RG 配置这样填：

| | `bank_npy_path` | `organ_annotation_path` |
|---|---|---|
| 训练 | `organ_report_embeddings.npz` | `train/organ_annotation.json` |
| 验证 | `organ_report_embeddings.npz`（**同一个**） | `val/organ_annotation.json` |

npz 和 json 的格式和以前完全一样，只是文本向量从 256 维变成 4096 维。Reg2RG 的读取代码不用改（RWLKE 会按 npz 自动设成 4096 维），但**模型要从头训练**。

**Reg2RG 的器官分支要用同一个 checkpoint 的 perceiver + fc。** bank 是按"ViT → perceiver → fc"这条路径检索出来的，Reg2RG 里器官裁剪图最好也走同一条路径。已经改好的版本在 `Reg2RG-debug-current-full_correct_F_20260920_1027`：器官分支单独用 `perceiver-<ckpt_name>.pth`（冻结）和 `fc-<ckpt_name>.pth`（继续训练），配置文件是 `configs/train_radgenome/perceiver_step145000.sh` 和 `configs/test_radgenome/perceiver_step145000.sh`。

## 流程图

从上往下按顺序执行。`(U)` `(L)` `(A)` `(P)` `(B)` `(M)` 是中间文件的代号，后面的步骤写"输入：(A)"就是用前面产出的 (A)；`★` 是最后要填进 Reg2RG 的文件。

### Train（`run_train.sh`）

```
┃ [0] get_unique_reports.py —— 报告去重
┃     输入：train 报告 csv（24126 人 × 10 个器官）
┃     做法：同一器官里一模一样的句子合并，记下每句话是哪些病人写的
┃           报告没提到的器官，补一句模板句 "<organ> shows no significant abnormalities."
┃     输出：(U) unique_reports.json
        │
        ▼
┃ [0b] label_bank.py —— 给每句话标正常/异常（新）
┃     输入：(U) + train_vqa_abnormality.csv
┃     做法：病人器官有不是 "no findings" 的发现 → 异常
┃           报告里写的是 "X was not detected" → 把 X 去掉（修正否定句）
┃           一句话的标签 = 写过它的病人多数票
┃     输出：(L) bank_labels.json
        │
        ▼
┃ [1] compute_text_embeddings_4096.py —— 句子变向量
┃     输入：(U)
┃     做法：每句话 → BERT → adapter → 4096 维向量
┃     输出：(A) organ_report_embeddings_unique_4096.npz
        │
        ▼
┃ [2] build_dense_bank.py —— 展开成每个病人一条（不变）
┃     输入：(U) + (A)
┃     输出：(B) ★ organ_report_embeddings.npz（24126 × 10 × 4096）
        │
        ▼
┃ [3] build_organ_annotation.py --split train —— 挑参考报告（改）
┃     输入：(A) (L) + train 的 CT 和器官 mask
┃     [1b] 先做原型（新）：每个器官，正常句子的平均向量 → 正常原型 Pn
┃                               异常句子的平均向量 → 异常原型 Pa
┃          （只用真实写出来的句子，不用模板句）→ (P) prototypes.npz
┃     a) CT 按 mask 裁剪 → ViT → perceiver → fc → 图像向量 q
┃     b) 零样本判断（不设阈值）：q 离 Pn 更近 → 正常；离 Pa 更近 → 异常
┃     c) 正常 → 只在正常句子里挑最近的 5 句
┃        异常 → 只在异常句子里挑最近的 5 句（图像和文本都先去均值）
┃        esophagus、trachea、breast、thyroid 不做判断 → 在所有句子里挑最近的 5 句
┃     d) 跳过同一个人所有扫描写过的句子
┃     输出：★ train/organ_annotation.json
┃           (P) prototypes.npz
┃           (M) train/image_mean.npz（train 图像均值，异常池去均值用）
┃           train/decisions.npz（零样本判断，检查用）、train/image_queries.pt（缓存）
```

| 步骤 | 做什么 |
|---|---|
| [0] | 把 train 报告按器官去重，记下每句话是哪些病人写的。 |
| **[0b]（新）** | 给每句话标上正常/异常，方法见下面"正常/异常怎么定义"。 |
| [1] | 每句话通过 BERT 和 `adapter` 变成 4096 维向量。 |
| [2] | 展开成 Reg2RG 读取的 bank（不变）。 |
| **[1b]（新，在 [3] 开头执行）** | 每个器官做两个原型：正常句子的平均向量 Pn、异常句子的平均向量 Pa。只用真实写出来的句子，不用模板句；不按写过这句话的人数加权（去掉模板句后，加不加权结果几乎一样，balanced acc 0.73 vs 0.74）。 |
| **[3]（改）** | a) CT 按器官 mask 裁剪，经过 ViT → `perceiver` → `fc` 得到图像向量 q。b) **零样本判断，不设阈值：** q 离正常原型更近就判为正常，否则判为异常。c) 判为正常的只在正常句子里挑最近的 5 句；判为异常的只在异常句子里挑，并且图像和文本都先去均值。esophagus、trachea、breast、thyroid 不做判断，直接在所有句子里挑。d) 跳过同一个人所有扫描写过的句子。另外存下 train 图像均值 (M)。 |

### Test（`run_test.sh`）

```
┃ 复用 train 的文件
┃     (U) (L) (A) (P) (B)：直接用 run_train.sh 生成的
┃     (M)：train 的 [3] 才会生成 → 必须先跑完 run_train.sh
        │
        ▼
┃ [3] build_organ_annotation.py --split val —— 挑参考报告
┃     输入：(A) (L) (P) (M) + val 的 CT 和器官 mask
┃     a) CT 按 mask 裁剪 → ViT → perceiver → fc → 图像向量 q
┃     b) 零样本判断（不设阈值）：离 Pn 更近 → 正常；离 Pa 更近 → 异常
┃     c) 正常 → 只在 train 的正常句子里挑最近的 5 句
┃        异常 → 只在 train 的异常句子里挑最近的 5 句（用 M 去均值）
┃        esophagus、trachea、breast、thyroid 不做判断 → 在所有句子里挑
┃     d) 不用排除同一个人（val 病人本来就不在 bank 里）
┃     输出：★ val/organ_annotation.json
┃           val/decisions.npz（检查用）、val/image_queries.pt（缓存）
```

| 步骤 | 做什么 |
|---|---|
| 前几步 | 全部复用 train 的文件。val 没有自己的 bank，候选都是 train 的句子。 |
| [3] | 判断和检索的方法和 train 完全一样；去均值用 train 存下的 (M)；不需要排除同一个人，因为 val 病人本来就不在 bank 里。 |

## 为什么要两阶段

旧的一阶段做法是在一个器官的所有句子里直接挑最近的 5 句。测试发现**正常的病人大多被配上了异常的报告**：全部 val 病人里，正常病人拿到的参考报告平均只有 25% 是正常的（abdomen 0%、lung 3%，见下面"两阶段的测试结果"）。

原因是**去重之后，异常句子的行数远多于正常句子**。正常的写法很集中（几千人共用几句话），异常的写法几乎每人一句：

| 器官 | 正常的 train 病人 | bank 里正常句子 | 异常句子 | 异常 : 正常 |
|---|---|---|---|---|
| breast | 97%（大多是模板句） | 203 | 628 | 3.1 倍 |
| bone | 66% | 1041 | 5502 | 5.3 倍 |
| pleura | 72% | 836 | 6417 | 7.7 倍 |
| lung | 18% | 824 | 19131 | 23.2 倍 |

（数字来自 `label_bank.py` 的输出，正常句子包含模板句那一行。）

即使一个正常病人"平均来说"离正常句子更近，从几千个异常句子里也总有几个恰好离他更近，挑最近的 5 句就会被数量压倒。所以先判断正常/异常，再只在同一类句子里挑。

## 正常/异常怎么定义

只用数据集自带的 `train_vqa_abnormality.csv`（RadGenome 从报告里自动抽取的"每次扫描、每个部位有哪些异常"）：

1. **病人的一个器官：** 有任何一项不是 "no findings" 的发现，就算异常；只写了 "no findings" 或者没有记录，算正常。
2. **否定句过滤：** 如果某个异常词在报告里后面紧跟 "was not detected / were not observed" 这类说法，就去掉这个词。这是因为 vqa 会把"未见心包积液"误抽成"心包积液"：**heart 有 58% 的"异常"其实是这种错误**，其他器官不到 1%。
3. **bank 里的一句话：** 由写过它的病人多数票决定。

局限：标签是自动抽取的，有噪声；有一点发现就算异常，不分轻重。

## 零样本判断

**怎么判断：** 每个器官有两个原型：正常原型 Pn、异常原型 Pa。图像向量 q 离 Pn 更近（cos(q, Pn) ≥ cos(q, Pa)）就判为正常，否则判为异常。**不设阈值**，不用任何数据集的统计量去调，所以换一批新病人、新机构也能直接用。

**原型只用真实写出来的句子。** 报告没提到的器官，我们会补一句模板句 "<organ> shows no significant abnormalities."，但它不参与原型：

- 训练时只用了报告里写了句子的器官，模板句从没出现过，所以它离所有图像都远；
- 如果用它做原型，breast、thyroid 的正常原型几乎就是这一句（99% 的人用它），所有人都会被判成异常；
- Reg2RG 训练时只用报告里写了句子的器官；测试时没写句子的器官也会查参考报告并生成文字，但它们的 GT 为空，逐区域评估会跳过，只影响整篇报告的指标（见下面"两阶段的测试结果" B）。

**这 4 个器官不做判断，直接在所有句子里挑：**

| 器官 | 原因 |
|---|---|
| esophagus | 分数有一定信息（AUC 0.70），但所有病人都离异常原型更近，不设阈值就会全部判成异常（判为正常 0%） |
| trachea and bronchie | 判断接近随机（AUC 0.56） |
| breast、thyroid | 报告里很少写这两个器官（val 只有约 50 人写了，约 3%），数据太少；thyroid 的标签也不可靠（见下面"注意"） |

**准不准：** 全部 1564 个 val 病人，只看这个器官写了句子的病人（标签用 vqa）。后 4 行是**假设**也给它们做判断补算的结果，实际生成 bank 时它们没有用判断。

- **准确率：** 判对的比例。正常/异常人数不均衡时会误导（trachea 全判正常也有 89%）；
- **balanced acc：** "正常病人判对率"和"异常病人判对率"的平均，随机猜约 0.5；
- **AUC：** 只看分数能不能把正常和异常病人排开，跟用 0 作分界无关，随机是 0.5。

| 器官 | 人数 | 真实正常 | 判为正常 | 准确率 | balanced acc | 正常病人判对 | 异常病人判对 | AUC | 实际用了判断 |
|---|---|---|---|---|---|---|---|---|---|
| abdomen | 1518 | 50% | 51% | 71% | 0.71 | 71% | 70% | 0.77 | 是 |
| bone | 1509 | 65% | 58% | 70% | 0.69 | 72% | 67% | 0.75 | 是 |
| heart | 1422 | 73% | 70% | 73% | 0.68 | 79% | 57% | 0.77 | 是 |
| lung | 1514 | 17% | 48% | 61% | 0.68 | 77% | 58% | 0.74 | 是 |
| mediastinum | 1516 | 58% | 47% | 71% | 0.72 | 65% | 79% | 0.77 | 是 |
| pleura | 1172 | 61% | 76% | 73% | 0.68 | 90% | 46% | 0.74 | 是 |
| esophagus | 1324 | 86% | 0% | 14% | 0.50 | 0% | 100% | 0.70 | 否 |
| trachea and bronchie | 1412 | 91% | 97% | 89% | 0.52 | 97% | 7% | 0.56 | 否 |
| breast | 56 | 16% | 11% | 84% | 0.63 | 33% | 94% | 0.64 | 否 |
| thyroid | 51 | 22% | 78% | 35% | 0.52 | 82% | 22% | 0.63 | 否 |

做判断的 6 个器官 balanced acc 平均 0.69（之前在 100 个 val 病人上测的是 0.74，小样本高估了）。lung 判为正常的偏多（48% vs 真实 17%），pleura 的异常病人只判对 46%。

（早先在 100 个 val 病人上比较过：不设阈值 balanced acc 0.74，用 train 按比例定阈值 0.70，所以选了不设阈值。）

## 两阶段的测试结果

全部 1564 个 val 病人，step145000，标签用修正否定句后的 vqa。

**A. 这个器官写了句子的病人**（Reg2RG 训练、逐区域评估用的就是这部分）

- **拿到正常参考：** 选出的 5 份参考报告里正常报告的比例；
- **finding 重合：** 只看异常病人，他的异常和参考报告的异常在关键词上的重合程度。

| 器官 | 人数 | 真实正常 | 判为正常 | 正常病人：拿到正常参考 | 异常病人：拿到正常参考 | 异常病人：finding 重合 |
|---|---|---|---|---|---|---|
| abdomen | 1518 | 50% | 51% | 0% → **71%** | 0% → 30% | 0.054 → 0.067 |
| bone | 1509 | 65% | 58% | 14% → **72%** | 4% → 33% | 0.086 → 0.098 |
| heart | 1422 | 73% | 70% | 39% → **79%** | 19% → 43% | 0.101 → 0.043 |
| lung | 1514 | 17% | 48% | 3% → **77%** | 3% → 42% | 0.081 → 0.062 |
| mediastinum | 1516 | 58% | 47% | 40% → **65%** | 13% → 21% | 0.104 → 0.115 |
| pleura | 1172 | 61% | 76% | 29% → **90%** | 17% → 54% | 0.126 → 0.106 |
| esophagus | 1324 | 86% | 不判断 | 19% | 10% | 0.607 |
| trachea and bronchie | 1412 | 91% | 不判断 | 1% | 1% | 0.051 |
| breast | 56 | 16% | 不判断 | 24% | 28% | 0.031 |
| thyroid | 51 | 22% | 不判断 | 84% | 81% | 0.008 |
| **平均** | | | | **25% → 58%** | 18% → 34% | 0.125 → 0.119 |

（箭头左边是旧的一阶段 `--mode plain`，右边是现在的做法；不判断的器官两者一样，只写一个数。）

**B. 这个器官没写句子的病人**（Reg2RG 测试时这些器官 GT 为空，逐区域评估跳过，只有整篇报告的 NLG 会看到它们的输出）：拿到正常参考的比例 breast（1323 人）25%，thyroid（1484 人）84%，pleura（392 人）24% → 78%，heart（131 人）33% → 69%，其他器官人数很少。

**结论：**

- **做判断的 6 个器官，正常病人明显改善：** 拿到正常参考的比例从 0–40% 提高到 65–90%。
- **代价：** 异常病人拿到正常参考的比例也上升了（判断只有中等准确，balanced acc 约 0.69）。lung 最明显：判为正常的偏多，异常病人有 42% 拿到正常参考，finding 重合从 0.081 降到 0.062。finding 重合整体平均基本不变（0.125 → 0.119）。
- **不做判断的器官，所有病人几乎都检索到同几句话（hubness）：** trachea 全部 val 病人的 7060 个参考位置只用到 39 句不同的话，前 4 句（都是支气管扩张/增厚之类的异常描述）占 77%，所以 91% 的正常病人几乎都拿到异常参考（1%）；thyroid 255 个位置只用到 11 句。thyroid 的 84% "正常"不是检索得好，而是这些句子被标成了正常（标签问题见下面"注意"）。

## 为什么 train 要跳过同一个人的所有扫描

scan id 的格式是 `train_10006_b_1`：`10006` 是**人**，`b` 是这个人的**第几次扫描**。train 里 24126 次扫描来自 19999 个人，其中 2637 人有不止一次扫描。

第 [3] 步给 train 扫描找参考报告时，会跳过**这个人所有扫描**写过的句子。比如查 `train_10006_a_1` 时，`a_1` 和 `b_1` 写过的句子都不会被选中（只对这个人跳过，别人照样能选到这些句子）。

**临床上医生参考病人以前的报告很正常，为什么这里算数据泄露？**

1. **这些"其他扫描"大多不是以前的检查，而是同一次检查。** 同一个人的不同扫描写了一字不差同一句报告的 (扫描, 器官) 有 11530 个，更可能是同一次检查重建出几个序列、共用一份报告。选到它，基本就是直接看到了答案。
2. **val 永远没有这个条件。** val 病人在 train 库里没有自己的扫描。如果 train 时能选到自己的扫描，KE 模块会学会"照抄参考报告"，到了 val 就没东西可抄。两边条件必须一致。

第 [3] 步最后会自动检查：train 的结果里没有任何一条来自同一个人，否则直接报错。

## 注意

- **两个脚本的 `ckpt_name` 必须一样。** 文字侧和图像侧是一起训出来的，混用不同 step 的结果没有意义。
- **必须先跑完 train 再跑 test。** test 要用 train 的图像均值 (M)，而且两个脚本共用前面几步的文件。
- **已经生成过的中间文件会自动跳过。** 想重新生成，就删掉对应的文件。
- **[3] 慢在哪里？** 慢在 CPU 读 CT、按 mask 裁剪器官，不在 GPU，所以多加 GPU 不会变快。要提速就调大脚本里的 `num_workers`（默认 16；每个约占 3–4GB 内存，机器是共享的，别开太大）。每个病人的 CT 只读一次、一次裁出所有器官，调用的是 Reg2RG 自己的 `mask_nii_img_to_tensor`，和 Reg2RG 的预处理完全一样（逐像素核对过）。
- **[3] 中断了可以接着跑吗？** 不能续跑，要从头开始。但如果图像已经编码完（`image_queries.pt` 已经存在），重跑只要几分钟。
- **bank 大约 4GB。** Reg2RG 会在每张 GPU 上放一份。`run_train.sh` 里的 `bank_dtype` 改成 `"float16"` 只能把**文件**减到约 2GB：Reg2RG 读取时会转回 float32（`my_embedding_layer.py` 里的 `.float()`），显存不会变少。改了要删掉旧的 `organ_report_embeddings.npz` 才会重新生成。
- **thyroid、breast 的数据很少。** 胸部 CT 报告一般只在甲状腺/乳腺有异常时才顺带写一句，train 只有约 4.5% 的扫描写了 thyroid（val 51 人），其余都是模板句。thyroid 的正常/异常标签也不可靠：vqa 表里 thyroid 只有 856 行，没有 vqa 行的病人默认算正常，所以 "Both lobes of the thyroid gland are slightly prominent and heterogeneous" 这样的句子也会被标成正常。
- **Reg2RG 的 KE 模块要用这个 4096 维 bank 重新训练**，用其他维度 bank 训出来的权重加载不了。

## config.py 是什么

`config.py` 不用单独跑，它是各个脚本**共用的设置文件**，里面放的是"用什么数据"：

| 内容 | 是什么 | 什么时候要改 |
|---|---|---|
| `DATA_ROOT` | 数据总目录 | 数据换了位置 |
| `SPLITS` | train / val 各自的 CT 目录、器官 mask 目录、报告 csv | 文件夹或 csv 改了名字 |
| `ORGANS` | 10 个器官的名字 | 一般不改 |
| `load_split()` | 读出一个 split 的**所有病人**，以及**每个病人每个器官用哪句报告**（csv 里没写这个器官的，用模板句 "<organ> shows no significant abnormalities."） | 一般不改 |

为什么要单独放一个文件：建文本库和做检索必须用**同一批病人、同一句报告**，否则跳过"同一个人的句子"时会对不上。

`config.py` 管"用什么数据"，`run_train.sh` / `run_test.sh` 管"用哪个 checkpoint、用哪张卡、结果存哪"。

## 文件说明

| 文件 | 作用 |
|---|---|
| `run_train.sh` / `run_test.sh` | 一步跑完，**平时只需要用这两个** |
| `config.py` | 共用设置（见上一节） |
| `get_unique_reports.py` | 第 [0] 步 |
| `label_bank.py` | 第 [0b] 步：给句子标正常/异常（新） |
| `compute_text_embeddings_4096.py` | 第 [1] 步 |
| `build_dense_bank.py` | 第 [2] 步 |
| `build_organ_annotation.py` | 第 [1b] 和 [3] 步，`--split train` 或 `--split val`；`--mode plain` 是旧的一阶段检索 |
