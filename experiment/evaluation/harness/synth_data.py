# -*- coding: utf-8 -*-
"""
合成中文记忆评测集生成器(确定性,种子固定)
=========================================
设计目标:给 SME 权重寻优提供有区分度的评测集。

特点:
1. 干扰项密集——同实体不同属性、同属性不同实体、反义偏好(喜欢↔讨厌)、
   "曾经…现在不…"时间反转,专测排序质量而不是肤浅词面匹配;
2. 时间戳分层(0天/30天/90天/180天/365天前)→ 让 freshness/recency/decay 权重可测;
3. importance 分层(0.2/0.5/0.8)→ 让 importance 权重可测;
4. 10% 英文混入 → 测跨语言行为;
5. 单 gold 与多 gold 查询并存 → hit@k / Recall@k / MRR / nDCG 都能算。

输出:datasets/synth/{name}.jsonl,每行一条记忆:
  {"id": "m000001", "text": "...", "ts": 1737000000, "importance": 0.5, "lang": "zh"}
查询集:datasets/synth/{name}.queries.jsonl,每行:
  {"qid": "q0001", "text": "...", "gold": ["m000001"], "type": "direct|temporal|multi"}
"""
import json
import random
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent / "datasets" / "synth"  # experiment/datasets/synth
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 实体池 ----------
SURNAMES = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤")
GIVEN = ["小明", "小红", "伟", "芳", "娜", "磊", "静", "强", "敏", "军", "洋", "艳", "杰", "涛", "明", "超", "霞", "平", "辉", "婷", "雪", "宇", "欣", "然", "子涵", "浩然", "雨萱", "梓萌", "一诺", "思远", "梦琪", "天佑", "佳怡", "睿", "恩", "皓", "岚", "澈", "煦", "望舒"]
FRUITS = ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "芒果", "草莓", "蓝莓", "樱桃", "荔枝", "龙眼", "菠萝", "猕猴桃", "桃子", "梨", "柿子", "石榴", "柚子", "椰子", "哈密瓜", "榴莲", "山竹", "杨梅", "枇杷", "橘子", "李子", "杏", "枣", "无花果", "百香果"]
COMPANIES = ["华为", "腾讯", "阿里巴巴", "字节跳动", "百度", "美团", "京东", "小米", "网易", "携程", "哔哩哔哩", "拼多多", "蚂蚁集团", "大疆", "宁德时代", "比亚迪", "理想汽车", "蔚来", "小鹏汽车", "商汤科技", "旷视科技", "科大讯飞", "紫光集团", "中芯国际", "京东方", "联想", "海尔", "格力", "顺丰", "中兴通讯"]
POSITIONS = ["软件工程师", "算法工程师", "产品经理", "项目经理", "数据分析师", "测试工程师", "运维工程师", "架构师", "前端开发", "后端开发", "安全工程师", "UI设计师", "运营专员", "市场经理", "人事主管", "财务经理", "法务顾问", "技术总监", "首席科学家", "实习生"]
TECH = ["Python", "Java", "Go", "Rust", "C++", "TypeScript", "Kubernetes", "Docker", "MySQL", "PostgreSQL", "Redis", "MongoDB", "Kafka", "Spark", "Flink", "PyTorch", "TensorFlow", "React", "Vue", "Spring Boot", "FastAPI", "gRPC", "Elasticsearch", "ClickHouse", "HBase", "Hadoop", "Ray", "LangChain", "向量数据库", "图数据库"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安", "重庆", "苏州", "天津", "长沙", "郑州", "青岛", "大连", "厦门", "合肥", "福州", "昆明", "贵阳", "南昌", "济南", "沈阳", "哈尔滨", "石家庄", "太原", "兰州", "乌鲁木齐", "拉萨"]
SPORTS = ["篮球", "足球", "羽毛球", "乒乓球", "网球", "游泳", "跑步", "骑行", "爬山", "瑜伽", "健身", "滑雪", "滑板", "攀岩", "拳击", "太极拳", "跳绳", "排球", "棒球", "高尔夫"]
DISHES = ["火锅", "烤肉", "寿司", "拉面", "饺子", "包子", "炒饭", "麻婆豆腐", "宫保鸡丁", "鱼香肉丝", "糖醋排骨", "红烧肉", "清蒸鱼", "白切鸡", "小笼包", "煎饼果子", "肉夹馍", "螺蛳粉", "麻辣烫", "披萨"]
HOBBIES = ["摄影", "绘画", "弹吉他", "弹钢琴", "下围棋", "看书", "写小说", "拼乐高", "养多肉", "钓鱼", "桌游", "剧本杀", "看纪录片", "收藏邮票", "做手冲咖啡", "烘焙", "剪视频", "写书法", "玩航模", "观鸟"]
MAJORS = ["计算机科学", "软件工程", "人工智能", "数据科学", "电子信息", "自动化", "数学", "物理学", "生物医学", "金融学", "会计学", "法学", "心理学", "建筑学", "土木工程", "机械工程", "材料科学", "环境工程", "汉语言文学", "新闻传播学"]

AGE_BUCKETS = [0, 7, 30, 90, 180, 365]          # 天前
IMP_BUCKETS = [0.2, 0.5, 0.8]

TEMPLATES = [
    # (模板, 查询模板, gold数, 类型)
    ("{p}喜欢{f}",                        "{p}喜欢什么水果",                       1, "direct"),
    ("{p}讨厌{f}",                        "{p}讨厌什么水果",                       1, "direct"),
    ("{p}在{c}担任{pos}",                 "{p}在哪家公司工作",                      1, "direct"),
    ("{p}在{c}的职位是{pos}",             "{p}是什么职位",                          1, "direct"),
    ("{p}住在{city}",                     "{p}住在哪个城市",                        1, "direct"),
    ("{p}周末喜欢{sport}",                "{p}周末做什么运动",                      1, "direct"),
    ("{p}最爱吃的菜是{dish}",             "{p}爱吃什么菜",                          1, "direct"),
    ("{p}的爱好是{hobby}",                "{p}有什么爱好",                          1, "direct"),
    ("{p}大学专业是{major}",              "{p}学的什么专业",                        1, "direct"),
    ("{p}的主要技术栈是{tech}",           "{p}用什么技术栈",                        1, "direct"),
    ("{p}曾经喜欢{f}，但现在更喜欢{f2}",   "{p}现在更喜欢什么水果",                   1, "temporal"),
    ("{p}以前住在{city}，上个月搬到了{city2}", "{p}现在住在哪个城市",                 1, "temporal"),
    ("{p}过去用{tech}，今年切换到了{tech2}",  "{p}现在用什么技术",                    1, "temporal"),
    ("{p}和{p2}在同一家公司{c}工作",       "{p}和谁是同事",                          2, "multi"),
    ("{p}和{p2}一起打{sport}",            "{p}和谁一起打球",                        2, "multi"),
    ("{city}的{p}是做{pos}的",            "{p}做什么工作",                          1, "direct"),
    ("{p}的生日是3月{d}号",               "{p}的生日是几月几号",                    1, "direct"),
    ("{p}养了一只叫{nick}的猫",           "{p}养了什么宠物",                        1, "direct"),
]
NICKS = ["豆豆", "团子", "橘子", "年糕", "布丁", "奶茶", "芝麻", "汤圆", "雪球", "煤球", "花卷", "馒头"]

EN_TEMPLATES = [
    ("{p} works as a {pos} at {c}",       "Where does {p} work",          1),
    ("{p} lives in {city}",               "Which city does {p} live in",  1),
    ("{p} prefers {tech} over others",    "What tech does {p} prefer",    1),
]


def gen_person(rnd):
    return rnd.choice(SURNAMES) + rnd.choice(GIVEN)


def build(n_mem, n_query, seed, en_ratio=0.1):
    rnd = random.Random(seed)
    now = int(time.time())
    mems, queries = [], []
    mid = [0]

    def new_id():
        mid[0] += 1
        return f"m{mid[0]:06d}"

    def emit(text, lang="zh"):
        m = {
            "id": new_id(),
            "text": text,
            "ts": now - int(rnd.choice(AGE_BUCKETS) * 86400),
            "importance": rnd.choice(IMP_BUCKETS),
            "lang": lang,
        }
        mems.append(m)
        return m["id"]

    # 去重集合防止同一句话生成两次
    seen_texts = set()
    while len(mems) < n_mem:
        if rnd.random() < en_ratio:
            t, qt, _ = rnd.choice(EN_TEMPLATES)
            p = gen_person(rnd)
            text = t.format(p=p, pos=rnd.choice(POSITIONS), c=rnd.choice(COMPANIES),
                            city=rnd.choice(CITIES), tech=rnd.choice(TECH))
            if text in seen_texts:
                continue
            seen_texts.add(text)
            gid = emit(text, "en")
            if rnd.random() < 0.35:
                queries.append({"qid": f"q{len(queries)+1:04d}",
                                "text": qt.format(p=p), "gold": [gid], "type": "direct"})
            continue
        t, qt, ngold, qtype = rnd.choice(TEMPLATES)
        p = gen_person(rnd); p2 = gen_person(rnd)
        slots = dict(p=p, p2=p2, f=rnd.choice(FRUITS), f2=rnd.choice(FRUITS),
                     c=rnd.choice(COMPANIES), pos=rnd.choice(POSITIONS),
                     city=rnd.choice(CITIES), city2=rnd.choice(CITIES),
                     sport=rnd.choice(SPORTS), dish=rnd.choice(DISHES),
                     hobby=rnd.choice(HOBBIES), major=rnd.choice(MAJORS),
                     tech=rnd.choice(TECH), tech2=rnd.choice(TECH),
                     d=rnd.randint(1, 28), nick=rnd.choice(NICKS))
        try:
            text = t.format(**slots)
            qtext = qt.format(**slots)
        except (KeyError, IndexError):
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        gid = emit(text)
        # 多 gold:同事/球友双向都是答案
        gold = [gid]
        if qtype == "multi":
            # 补出配对句作为第二 gold(如果没生成过)
            alt = t.format(**{**slots, "p": p2, "p2": p})
            if alt not in seen_texts and rnd.random() < 0.9:
                seen_texts.add(alt)
                gold.append(emit(alt))
        # 抽样生成查询(保证查询数可控)
        if len(queries) < n_query and rnd.random() < 0.5:
            queries.append({"qid": f"q{len(queries)+1:04d}",
                            "text": qtext, "gold": gold, "type": qtype})

    return mems, queries


def dump(name, mems, queries):
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for m in mems:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(OUT / f"{name}.queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[{name}] memories={len(mems)} queries={len(queries)}")


if __name__ == "__main__":
    # dev: 调试用小集 / small / medium / large / xl(超级测)
    for name, n_mem, n_query, seed in [
        ("dev", 300, 80, 42),
        ("small", 1000, 250, 1001),
        ("medium", 5000, 1000, 2002),
        ("large", 20000, 2500, 3003),
        ("xl", 60000, 3000, 4004),
    ]:
        mems, queries = build(n_mem, n_query, seed)
        dump(name, mems, queries)
    print("done ->", OUT)
