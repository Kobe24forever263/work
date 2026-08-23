# -*- coding: utf-8 -*-
"""基于 0806进展与问题汇报.md 生成项目进展汇报 PPT（白底黑字、简洁风格）。"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import copy

# ---------- 基础配置 ----------
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.6)

# 颜色（以黑/灰为主，少量状态色）
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK = RGBColor(0x1F, 0x1F, 0x1F)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
MIDGRAY = RGBColor(0xD9, 0xD9, 0xD9)
BORDER = RGBColor(0xBF, 0xBF, 0xBF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
# 状态色（按文档规范：绿=已通过，黄=待优化，蓝=下一步，橙=货物）
GREEN = RGBColor(0x2E, 0x8B, 0x57)
YELLOW = RGBColor(0xB8, 0x86, 0x0B)
BLUE = RGBColor(0x1F, 0x6F, 0xB2)
ORANGE = RGBColor(0xE8, 0x6A, 0x17)

FONT = "Microsoft YaHei"

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def set_font(run, size, bold=False, color=BLACK, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    # 设置中文字体
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {})
        rPr.append(ea)
    ea.set('typeface', name)


def add_textbox(slide, x, y, w, h, lines, anchor=MSO_ANCHOR.TOP):
    """lines: list of (text, size, bold, color) or list of list-of-paragraphs."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    first = True
    for para in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = para.get('align', PP_ALIGN.LEFT)
        p.space_after = Pt(para.get('space_after', 6))
        p.line_spacing = para.get('line_spacing', 1.1)
        for seg in para['runs']:
            r = p.add_run()
            r.text = seg[0]
            set_font(r, seg[1], seg[2] if len(seg) > 2 else False,
                     seg[3] if len(seg) > 3 else BLACK)
    return tb


def add_rect(slide, x, y, w, h, fill=None, line=None, line_w=1.0, shape=MSO_SHAPE.RECTANGLE, shadow=False):
    sp = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    return sp


def rect_text(sp, lines, anchor=MSO_ANCHOR.MIDDLE, ml=0.12):
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(ml)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    first = True
    for para in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = para.get('align', PP_ALIGN.CENTER)
        p.space_after = Pt(para.get('space_after', 2))
        p.line_spacing = para.get('line_spacing', 1.05)
        for seg in para['runs']:
            r = p.add_run()
            r.text = seg[0]
            set_font(r, seg[1], seg[2] if len(seg) > 2 else False,
                     seg[3] if len(seg) > 3 else BLACK)


def new_slide():
    s = prs.slides.add_slide(BLANK)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=WHITE)
    return s


def add_title(slide, title, subtitle=None, num=None):
    """统一内容页标题：左侧编号 + 标题 + 底部细线。"""
    x = MARGIN
    y = Inches(0.42)
    if num:
        add_rect(slide, x, y + Inches(0.08), Inches(0.16), Inches(0.52), fill=BLACK)
        add_textbox(slide, x + Inches(0.34), y, Inches(1.0), Inches(0.7),
                    [{'runs': [(num, 22, True, GRAY)], 'align': PP_ALIGN.LEFT}])
        tx = x + Inches(0.34) + Inches(1.0)
    else:
        tx = x
    add_textbox(slide, tx, y, SLIDE_W - tx - MARGIN, Inches(0.7),
                [{'runs': [(title, 26, True, BLACK)]}])
    if subtitle:
        add_textbox(slide, tx, y + Inches(0.56), SLIDE_W - tx - MARGIN, Inches(0.4),
                    [{'runs': [(subtitle, 13, False, GRAY)]}])
    add_rect(slide, x, y + Inches(0.98), SLIDE_W - 2 * MARGIN, Pt(2.2), fill=MIDGRAY)


def add_footer(slide, page_no, text="双层仓储异构机器人协作运输 · 2026-08-06"):
    add_textbox(slide, MARGIN, SLIDE_H - Inches(0.42), Inches(8), Inches(0.3),
                [{'runs': [(text, 9, False, GRAY)]}])
    add_textbox(slide, SLIDE_W - MARGIN - Inches(1), SLIDE_H - Inches(0.42), Inches(1), Inches(0.3),
                [{'runs': [(f"{page_no:02d}", 10, True, GRAY)], 'align': PP_ALIGN.RIGHT}])


def add_status_tag(slide, x, y, text, color, w=Inches(1.35), h=Inches(0.32), size=11):
    sp = add_rect(slide, x, y, w, h, fill=color, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    rect_text(sp, [{'runs': [(text, size, True, WHITE)]}])
    return sp


def bullets(items, size=15, gap=8, indent=False, bullet_char="•"):
    paras = []
    for it in items:
        if isinstance(it, tuple) and len(it) == 2 and isinstance(it[1], RGBColor):
            text, color = it
        elif isinstance(it, tuple):
            text = it[0]  # (text, int) 形式，忽略数字
            color = BLACK
        else:
            text, color = it, BLACK
        runs = [(f"{bullet_char} ", size, True, GRAY), (text, size, False, color)]
        paras.append({'runs': runs, 'space_after': gap, 'line_spacing': 1.15})
    return paras


# ============================================================
# 第1页：封面
# ============================================================
s = new_slide()
add_rect(s, 0, 0, SLIDE_W, Inches(0.28), fill=BLACK)
add_rect(s, 0, SLIDE_H - Inches(0.28), SLIDE_W, Inches(0.28), fill=BLACK)
# 顶部小字
add_textbox(s, MARGIN, Inches(1.15), SLIDE_W - 2 * MARGIN, Inches(0.4),
            [{'runs': [("项目阶段汇报 · Phase Report", 14, False, GRAY)], 'align': PP_ALIGN.CENTER}])
# 主标题
add_textbox(s, MARGIN, Inches(2.1), SLIDE_W - 2 * MARGIN, Inches(1.4),
            [{'runs': [("双层仓储异构机器人协作运输", 40, True, BLACK)], 'align': PP_ALIGN.CENTER, 'space_after': 4},
             {'runs': [("项目进展与问题汇报", 40, True, BLACK)], 'align': PP_ALIGN.CENTER}])
# 副标题
add_textbox(s, MARGIN, Inches(3.9), SLIDE_W - 2 * MARGIN, Inches(0.5),
            [{'runs': [("Carter + Go2 + SCAN-Planner + 事件驱动 SMDP", 18, False, DARK)], 'align': PP_ALIGN.CENTER}])
# 日期
add_textbox(s, MARGIN, Inches(4.75), SLIDE_W - 2 * MARGIN, Inches(0.5),
            [{'runs': [("汇报日期：2026 年 8 月 6 日", 15, False, GRAY)], 'align': PP_ALIGN.CENTER}])
# 底部标签
tags = [("阶段 11 · 已验收", GREEN), ("等待性能 · 待优化", YELLOW), ("阶段 12 · 下一步", BLUE)]
tw = Inches(2.3)
x0 = (SLIDE_W - len(tags) * tw - (len(tags) - 1) * Inches(0.3)) / 2
for i, (t, c) in enumerate(tags):
    add_status_tag(s, x0 + i * (tw + Inches(0.3)), Inches(5.6), t, c, w=tw, h=Inches(0.42), size=12)

# ============================================================
# 第2页：项目目标与场景
# ============================================================
s = new_slide()
add_title(s, "项目目标与场景", "双层仓储环境 · 异构机器人 · 同层与跨楼层运输", "02")
add_status_tag(s, SLIDE_W - MARGIN - Inches(1.9), Inches(0.5), "场景目标", BLUE, w=Inches(1.9))

# 目标一句话（顶部框）
goal = add_rect(s, MARGIN, Inches(1.35), SLIDE_W - 2 * MARGIN, Inches(1.05), fill=LIGHT, line=BORDER)
rect_text(goal, [
    {'runs': [("目标：", 16, True, BLACK), ("完成同层与跨楼层货物运输，并为强化学习调度建立统一、可学习的环境。", 16, False, BLACK)]}
], anchor=MSO_ANCHOR.MIDDLE)

# 左：系统组成
add_textbox(s, MARGIN, Inches(2.7), Inches(3.0), Inches(0.4),
            [{'runs': [("系统组成", 16, True, BLACK)]}])
add_rect(s, MARGIN, Inches(3.15), Inches(3.0), Pt(2), fill=MIDGRAY)
add_textbox(s, MARGIN, Inches(3.35), Inches(5.9), Inches(3.6),
            bullets([
                ("2 辆 Carter 轮式小车（一楼 / 二楼各一辆）", 14),
                ("4 只 Go2 四足机器人", 14),
                ("4 条跨楼层楼梯", 14),
                ("装卸货平台、交接点、待命泊位", 14),
            ], size=14, gap=12))

# 右：统一技术路线
rx = Inches(7.0)
add_textbox(s, rx, Inches(2.7), Inches(4.5), Inches(0.4),
            [{'runs': [("统一技术路线", 16, True, BLACK)]}])
add_rect(s, rx, Inches(3.15), Inches(4.5), Pt(2), fill=MIDGRAY)
add_textbox(s, rx, Inches(3.35), Inches(5.7), Inches(3.6),
            bullets([
                ("Carter 与 Go2 均使用 SCAN-Planner 执行路径", 13.5),
                ("高层调度器只选任务 / 机器人 / 楼梯 / 任务后策略", 13.5),
                ("使用既定确定性路线，不加入随机扰动", 13.5),
                ("RViz 做功能和路线验收，Gazebo 收尾阶段统一优化", 13.5),
                ("任务之间不整体复位，状态跨任务保留", 13.5),
            ], size=13.5, gap=10))
add_footer(s, 2)

# ============================================================
# 第3页：完整跨楼层任务链
# ============================================================
s = new_slide()
add_title(s, "完整跨楼层任务链", "六步水平流程 · 橙色方块表示货物所有权在三种载体之间切换", "03")

steps = [
    ("1 一楼取货", "起点 Carter\n前往装货平台取货", "Carter"),
    ("2 Carter 运送", "运送到所选楼梯\n发送交接点", "Carter"),
    ("3 Go2 交接上楼", "货物切换并吸附\nGo2 携货通过楼梯", "Go2"),
    ("4 二楼交接", "Go2 走出楼梯口\n与二楼 Carter 交接", "Carter"),
    ("5 Carter 送货", "送至目标平台\n并卸货", "Carter"),
    ("6 泊位待命", "进入互不重叠的\n空闲待命泊位", "全体"),
]
n = len(steps)
gap = Inches(0.18)
bw = (SLIDE_W - 2 * MARGIN - (n - 1) * gap) / n
bh = Inches(2.1)
by = Inches(2.2)
for i, (title, desc, carrier) in enumerate(steps):
    x = MARGIN + i * (bw + gap)
    sp = add_rect(s, x, by, bw, bh, fill=WHITE, line=BLACK, line_w=1.2)
    # 顶部标题条
    hd = add_rect(s, x, by, bw, Inches(0.5), fill=BLACK)
    rect_text(hd, [{'runs': [(title, 13, True, WHITE)]}])
    # 描述
    rect_text(sp, [
        {'runs': [(desc, 11.5, False, DARK)], 'line_spacing': 1.15}
    ], anchor=MSO_ANCHOR.MIDDLE)
    # 货物橙色方块 + 载体标签
    if i < 5:
        add_rect(s, x + bw / 2 - Inches(0.14), by + bh + Inches(0.22), Inches(0.28), Inches(0.28), fill=ORANGE)
        add_textbox(s, x - Inches(0.1), by + bh + Inches(0.55), bw + Inches(0.2), Inches(0.4),
                    [{'runs': [(carrier, 12, True, BLACK)], 'align': PP_ALIGN.CENTER}])
    else:
        add_rect(s, x + bw / 2 - Inches(0.14), by + bh + Inches(0.22), Inches(0.28), Inches(0.28), fill=MIDGRAY)
        add_textbox(s, x - Inches(0.1), by + bh + Inches(0.55), bw + Inches(0.2), Inches(0.4),
                    [{'runs': [("空闲泊位", 12, True, BLACK)], 'align': PP_ALIGN.CENTER}]
                   )
    if i < n - 1:
        ar = add_rect(s, x + bw + Inches(0.02), by + bh / 2 - Inches(0.18), gap - Inches(0.04), Inches(0.36),
                      fill=BLACK, shape=MSO_SHAPE.RIGHT_ARROW)

# 底部图例
add_textbox(s, MARGIN, Inches(5.6), SLIDE_W - 2 * MARGIN, Inches(1.2),
            [{'runs': [("橙色方块 = 货物；", 13, False, BLACK),
                       ("载体依次为：Carter → Go2 → Carter → 空闲泊位", 13, False, BLACK)]},
             {'runs': [("注：货物采用可视化方块与逻辑吸附，不模拟重力，交接时允许瞬时转移；Go2 在楼梯口少量距离后交货。", 12, False, GRAY)]}])
add_footer(s, 3)

# ============================================================
# 第4页：当前系统架构
# ============================================================
s = new_slide()
add_title(s, "当前系统架构", "分层架构 · 高层策略不直接输出速度 / 关节 / 步态", "04")

layers = [
    ("高层 · 事件驱动调度器 / SMDP", [
        "选择任务、机器人组合、楼梯、任务后策略",
        "不直接控制速度、关节或步态",
        "任务异步到达并排队，阶段11按串行执行",
    ], BLACK),
    ("中层 · 任务 / 货物 / 资源锁 / 交接状态机", [
        "任务队列与动作掩码",
        "货物逻辑吸附与所有权切换",
        "区域、交接点、楼梯的占用 / 释放审计",
    ], DARK),
    ("执行层 · SCAN-Planner", [
        "Carter 与 Go2 均使用 SCAN-Planner 执行具体路径",
        "两者运动能力模型不同，共享高层任务调度",
    ], DARK),
    ("显示层 · RViz", [
        "仓库三维点云、占据区域、规划轨迹、机器人标签",
        "Go2 / Carter 模型统一显示，用于功能和路线验收",
    ], DARK),
]
ly = Inches(1.5)
lh = Inches(1.28)
lgap = Inches(0.14)
for i, (name, items, color) in enumerate(layers):
    y = ly + i * (lh + lgap)
    # 左侧名称块
    nb = add_rect(s, MARGIN, y, Inches(3.7), lh, fill=BLACK)
    rect_text(nb, [{'runs': [(name, 14, True, WHITE)], 'line_spacing': 1.1}], anchor=MSO_ANCHOR.MIDDLE)
    # 右侧内容
    cb = add_rect(s, MARGIN + Inches(3.7) + Inches(0.15), y, SLIDE_W - 2 * MARGIN - Inches(3.85), lh,
                  fill=WHITE, line=BORDER)
    rect_text(cb, bullets(items, size=12.5, gap=3), anchor=MSO_ANCHOR.MIDDLE)
    # 向下箭头
    if i < len(layers) - 1:
        add_rect(s, MARGIN + Inches(1.6), y + lh + Inches(0.015), Inches(0.5), lgap - Inches(0.03),
                 fill=BORDER, shape=MSO_SHAPE.DOWN_ARROW)

add_textbox(s, MARGIN, ly + 4 * (lh + lgap) + Inches(0.15), SLIDE_W - 2 * MARGIN, Inches(0.5),
            [{'runs': [("要点：", 14, True, BLACK),
                       ("高层只做任务级决策；路径执行统一交给 SCAN-Planner；低层运动控制（速度 / 关节 / 步态）不在当前调度范围内。", 14, False, DARK)]}])
add_footer(s, 4)

# ============================================================
# 第5页：已完成的场景与规划工作
# ============================================================
s = new_slide()
add_title(s, "已完成的场景与规划工作", "双层仓库与基础场景 · RViz 显示 · 路线规划与协作运输", "05")

left = [
    ("双层仓库与基础场景", [
        "建立双层仓库、中央镂空区、四条楼梯、护栏",
        "修正楼梯顶部与二楼平台连接（避免最后一级掉落）",
        "一楼照明 + 5 根二楼支撑柱，楼梯黄色材质",
        "障碍区改造为装卸 / 运输平台，一楼外围航线扩展至楼梯交接区",
        "4 只 Go2 与多辆 Carter 纳入统一仓储任务场景",
    ]),
]
right = [
    ("RViz / 模型 / 点云显示", [
        "生成 SCAN-Planner 用 PCD 点云与 YAML 配置",
        "修复 Go2 RobotModel 因 TF 链不完整导致的爆红问题",
        "点云、占据区、轨迹、标签、Go2 / Carter 模型统一显示",
        "演示必须显示机器人本体，不能只显示路径 / 标签",
    ]),
    ("路线规划与协作运输", [
        "Carter 由 Nav2 统一切换为 SCAN-Planner，直线行驶效果",
        "验证一楼取货 → 交接 → Go2 上楼 → 二楼接货 → 卸货完整路线",
        "验证其余楼梯同类交接路线，楼梯不绑定固定象限",
        "Go2 与 Carter 共享高层任务调度可行",
    ]),
]

def render_panel(s, x, y, w, heading, items):
    add_rect(s, x, y, w, Inches(0.5), fill=BLACK)
    add_textbox(s, x + Inches(0.15), y + Inches(0.08), w - Inches(0.3), Inches(0.4),
                [{'runs': [(heading, 14, True, WHITE)]}])
    add_textbox(s, x + Inches(0.1), y + Inches(0.66), w - Inches(0.2), Inches(4.4),
                bullets(items, size=12.5, gap=9))

col_w = (SLIDE_W - 2 * MARGIN - Inches(0.3)) / 2
render_panel(s, MARGIN, Inches(1.45), col_w, "① 双层仓库与基础场景", left[0][1])
render_panel(s, MARGIN + col_w + Inches(0.3), Inches(1.45), col_w, "② RViz / 模型 / 点云显示", right[0][1])
render_panel(s, MARGIN, Inches(4.55), col_w, "③ 路线规划与协作运输", right[1][1])

add_textbox(s, MARGIN + col_w + Inches(0.3), Inches(4.55), col_w, Inches(2.0),
            [{'runs': [("结论：", 14, True, GREEN),
                       ("场景、显示与四楼梯协作路线均已就绪，进入阶段 11 正式验收。", 14, False, DARK)]}])
add_footer(s, 5)

# ============================================================
# 第6页：货物与交接可视化
# ============================================================
s = new_slide()
add_title(s, "货物与交接可视化", "货物显示、吸附、转移与卸货 · 已人工验收通过", "06")

add_status_tag(s, SLIDE_W - MARGIN - Inches(2.1), Inches(0.5), "已人工验收", GREEN, w=Inches(2.1))

# 交接链
chain = [
    ("平台", "货物位于装卸货平台\n（有一定高度，非地面）"),
    ("Carter", "取货并吸附\n（缩小方块靠近本体）"),
    ("Go2", "Carter→Go2\n瞬时切换吸附"),
    ("Carter", "Go2→二楼 Carter\n交接"),
    ("平台", "目标平台卸货\n所有权切换"),
]
n = len(chain)
gap = Inches(0.18)
bw = (SLIDE_W - 2 * MARGIN - (n - 1) * gap) / n
by = Inches(2.1)
bh = Inches(2.2)
for i, (who, desc) in enumerate(chain):
    x = MARGIN + i * (bw + gap)
    box = add_rect(s, x, by, bw, bh, fill=WHITE, line=BLACK, line_w=1.2)
    # 顶部
    hd = add_rect(s, x, by, bw, Inches(0.5), fill=BLACK)
    rect_text(hd, [{'runs': [("承运载体", 12, True, WHITE)]}])
    # 橙色方块/载体图标
    sq = add_rect(s, x + bw / 2 - Inches(0.18), by + Inches(0.72), Inches(0.36), Inches(0.36),
                  fill=ORANGE if i in (0, 4) else MIDGRAY)
    if i in (0, 4):
        add_textbox(s, x, by + Inches(1.1), bw, Inches(0.3),
                    [{'runs': [(who, 12, True, BLACK)], 'align': PP_ALIGN.CENTER}])
    else:
        add_textbox(s, x, by + Inches(1.1), bw, Inches(0.3),
                    [{'runs': [(who, 12, True, BLACK)], 'align': PP_ALIGN.CENTER}])
    rect_text(box, [{'runs': [(desc, 10.5, False, DARK)], 'line_spacing': 1.12}], anchor=MSO_ANCHOR.BOTTOM,
              ml=0.06)
    add_rect(s, x, by + Inches(1.42), bw, Pt(1.2), fill=MIDGRAY)
    if i < n - 1:
        add_rect(s, x + bw + Inches(0.02), by + bh / 2 - Inches(0.18), gap - Inches(0.04), Inches(0.36),
                 fill=BLACK, shape=MSO_SHAPE.RIGHT_ARROW)

# 三条关键标注
notes = [
    ("无重力逻辑吸附", "货物为可视化方块 + 逻辑吸附，不模拟货物重力"),
    ("交接瞬时切换", "交接完成无需额外长时间停顿，确认吸附成功即继续"),
    ("逻辑层验收通过", "货物显示 / 吸附 / 转移 / 卸货已人工验收，不再作为当前主要风险"),
]
nw = (SLIDE_W - 2 * MARGIN - 2 * Inches(0.25)) / 3
for i, (t, d) in enumerate(notes):
    x = MARGIN + i * (nw + Inches(0.25))
    c = add_rect(s, x, Inches(4.85), nw, Inches(1.5), fill=LIGHT, line=BORDER)
    add_textbox(s, x + Inches(0.15), Inches(5.0), nw - Inches(0.3), Inches(0.4),
                [{'runs': [(t, 13, True, BLACK)]}])
    add_textbox(s, x + Inches(0.15), Inches(5.42), nw - Inches(0.3), Inches(0.85),
                [{'runs': [(d, 11.5, False, DARK)], 'line_spacing': 1.15}])
add_footer(s, 6)

# ============================================================
# 第7页：阶段11做了什么
# ============================================================
s = new_slide()
add_title(s, "阶段 11 · 持久状态与事件驱动调度", "已正式验收并封板，形成规则与强化学习共用的事件驱动 SMDP 环境", "07")

cards = [
    ("无损异步任务队列", "策略每次观察优先级最高的 K=8 个任务"),
    ("四类任务", "一楼同层 / 二楼同层 / 跨层上行 / 跨层下行"),
    ("动态组合选择", "动态选择取货 Carter、Go2、楼梯、接货 Carter"),
    ("持久状态", "位置 / 楼层 / 累计距离 / 任务数 / 上一任务结果跨任务保留"),
    ("task_reset 局部清理", "只清理本任务状态，机器人不整体回到初始位置"),
    ("唯一空闲泊位", "同楼层、同类型且互不重叠的最近空闲泊位"),
    ("动作掩码", "拒绝错误楼层 / 类型 / 忙碌 / 故障 / 已有任务 / 带货机器人"),
    ("资源占用 / 释放审计", "区域、交接点、楼梯真实占用与释放事件"),
    ("SMDP 回报", "按异步任务实际到达时刻积分未完成任务时间"),
    ("完整决策记录", "state / action / mask / reward / next_state / terminated 全保存"),
]
rows, cols = 2, 5
cw = (SLIDE_W - 2 * MARGIN - (cols - 1) * Inches(0.18)) / cols
ch = Inches(2.05)
cgap = Inches(0.18)
for idx, (t, d) in enumerate(cards):
    r, c = divmod(idx, cols)
    x = MARGIN + c * (cw + cgap)
    y = Inches(1.55) + r * (ch + cgap)
    box = add_rect(s, x, y, cw, ch, fill=WHITE, line=BORDER)
    add_rect(s, x, y, cw, Inches(0.09), fill=BLACK)
    add_textbox(s, x + Inches(0.12), y + Inches(0.2), cw - Inches(0.24), Inches(0.6),
                [{'runs': [(t, 12.5, True, BLACK)], 'line_spacing': 1.05}])
    add_textbox(s, x + Inches(0.12), y + Inches(0.85), cw - Inches(0.24), ch - Inches(1.0),
                [{'runs': [(d, 10.5, False, GRAY)], 'line_spacing': 1.15}])

add_textbox(s, MARGIN, Inches(6.25), SLIDE_W - 2 * MARGIN, Inches(0.6),
            [{'runs': [("回报要点：", 14, True, BLACK),
                       ("按异步任务的实际到达时刻积分未完成任务时间作为 SMDP 回报。", 14, False, DARK)]}])
add_footer(s, 7)

# ============================================================
# 第8页：阶段11正式验收结果
# ============================================================
s = new_slide()
add_title(s, "阶段 11 正式验收结果", "medium 到达负载 · 固定种子 20260805 · 同种子完整运行两遍", "08")

add_status_tag(s, SLIDE_W - MARGIN - Inches(2.9), Inches(0.5), "阶段11纯逻辑环境正式验收通过", GREEN, w=Inches(2.9), size=11)

# 大数字卡片
nums = [
    ("20 / 20", "连续任务 PASS", GREEN),
    ("17 / 17", "核心自动测试", GREEN),
    ("14 种", "动态协作组合", GREEN),
    ("40 / 40", "非法动作被拒绝", GREEN),
    ("120 条", "资源占用/释放事件", GREEN),
    ("0 残留", "资源/货物/任务/位置重叠", GREEN),
]
nw = (SLIDE_W - 2 * MARGIN - 2 * Inches(0.2)) / 3
nh = Inches(1.15)
for i, (num, label, color) in enumerate(nums):
    r, c = divmod(i, 3)
    x = MARGIN + c * (nw + Inches(0.2))
    y = Inches(1.5) + r * (nh + Inches(0.16))
    box = add_rect(s, x, y, nw, nh, fill=WHITE, line=BORDER)
    add_rect(s, x, y, Inches(0.09), nh, fill=color)
    add_textbox(s, x + Inches(0.22), y + Inches(0.1), nw - Inches(0.3), Inches(0.6),
                [{'runs': [(num, 24, True, color)], 'align': PP_ALIGN.LEFT}])
    add_textbox(s, x + Inches(0.22), y + Inches(0.72), nw - Inches(0.3), Inches(0.4),
                [{'runs': [(label, 12, False, DARK)], 'align': PP_ALIGN.LEFT}])

# 四类任务各5次
qy = Inches(4.35)
add_textbox(s, MARGIN, qy, Inches(4), Inches(0.4),
            [{'runs': [("四类任务：各 5 次", 14, True, BLACK)]}])
quads = [("一楼同层", 5), ("二楼同层", 5), ("跨层上行", 5), ("跨层下行", 5)]
qw = (SLIDE_W - 2 * MARGIN - 3 * Inches(0.2)) / 4
for i, (name, cnt) in enumerate(quads):
    x = MARGIN + i * (qw + Inches(0.2))
    q = add_rect(s, x, qy + Inches(0.45), qw, Inches(0.9), fill=LIGHT, line=BORDER)
    rect_text(q, [
        {'runs': [(name, 12.5, False, DARK)]},
        {'runs': [(f"{cnt} 次", 18, True, BLACK)]}
    ])

add_textbox(s, MARGIN, Inches(6.0), SLIDE_W - 2 * MARGIN, Inches(0.7),
            [{'runs': [("仿真总时间 1408.04s · 平均任务时长 70.00s · 平均等待 296.22s · 队列峰值 10 · 同种子重放完全一致",
                        13, False, GRAY)]}])
add_footer(s, 8)

# ============================================================
# 第9页：机器人状态持续与泊位策略
# ============================================================
s = new_slide()
add_title(s, "机器人状态持续与泊位策略", "任务结束位置 → 最近空闲泊位 → 下一任务从该位置继续", "09")

# 状态流
flow = [
    ("任务结束", "当前任务完成\n位置 / 状态保留"),
    ("最近空闲泊位", "同楼层同类型\n互不重叠"),
    ("下一任务", "从上一任务结束的\n真实逻辑状态继续"),
]
fw = (SLIDE_W - 2 * MARGIN - 2 * Inches(1.1)) / 3
fy = Inches(2.0)
fh = Inches(1.7)
for i, (t, d) in enumerate(flow):
    x = MARGIN + i * (fw + Inches(1.1))
    box = add_rect(s, x, fy, fw, fh, fill=WHITE, line=BLACK, line_w=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    rect_text(box, [
        {'runs': [(t, 15, True, BLACK)]},
        {'runs': [(d, 11.5, False, DARK)], 'space_after': 0}
    ])
    if i < 2:
        add_rect(s, x + fw + Inches(0.05), fy + fh / 2 - Inches(0.22), Inches(1.0), Inches(0.44),
                 fill=BLACK, shape=MSO_SHAPE.RIGHT_ARROW)

# 三要点
pts = [
    ("不重置", "task_reset 只清理本任务状态，机器人不整体回到初始位置。"),
    ("泊位唯一", "泊位在逻辑层保证唯一且互不重叠，20 任务门禁证明无相同最终坐标。"),
    ("作业点及时释放", "完成任务后进入泊位，作业区 / 交接点 / 楼梯资源及时释放。"),
]
pw = (SLIDE_W - 2 * MARGIN - 2 * Inches(0.25)) / 3
for i, (t, d) in enumerate(pts):
    x = MARGIN + i * (pw + Inches(0.25))
    c = add_rect(s, x, Inches(4.35), pw, Inches(1.7), fill=LIGHT, line=BORDER)
    add_textbox(s, x + Inches(0.15), Inches(4.5), pw - Inches(0.3), Inches(0.4),
                [{'runs': [(t, 14, True, BLACK)]}])
    add_textbox(s, x + Inches(0.15), Inches(4.95), pw - Inches(0.3), Inches(1.0),
                [{'runs': [(d, 11.5, False, DARK)], 'line_spacing': 1.2}])

add_status_tag(s, MARGIN, Inches(6.3), "注意：逻辑泊位 ≠ Gazebo 物理验收", YELLOW, w=Inches(3.4), size=12)
add_footer(s, 9)

# ============================================================
# 第10页：主要问题——等待时间
# ============================================================
s = new_slide()
add_title(s, "主要问题 · Medium 负载下等待时间较长", "当前最重要性能问题，但非死锁、非任务卡死", "10")
add_status_tag(s, SLIDE_W - MARGIN - Inches(1.9), Inches(0.5), "性能瓶颈", YELLOW, w=Inches(1.9))

nums = [
    ("296.22 s", "平均等待时间"),
    ("558.29 s", "P95 等待时间"),
    ("782.36 s", "最大等待时间"),
    ("10", "队列峰值"),
]
nw = (SLIDE_W - 2 * MARGIN - 3 * Inches(0.25)) / 4
for i, (num, label) in enumerate(nums):
    x = MARGIN + i * (nw + Inches(0.25))
    box = add_rect(s, x, Inches(1.5), nw, Inches(1.4), fill=WHITE, line=BORDER)
    add_rect(s, x, y if False else Inches(1.5), nw, Inches(0.09), fill=YELLOW)
    add_textbox(s, x + Inches(0.15), Inches(1.7), nw - Inches(0.3), Inches(0.6),
                [{'runs': [(num, 22, True, YELLOW)], 'align': PP_ALIGN.CENTER}])
    add_textbox(s, x + Inches(0.15), Inches(2.4), nw - Inches(0.3), Inches(0.4),
                [{'runs': [(label, 12, False, DARK)], 'align': PP_ALIGN.CENTER}])

# 原因
add_textbox(s, MARGIN, Inches(3.2), Inches(6), Inches(0.4),
            [{'runs': [("原因判断", 15, True, BLACK)]}])
add_rect(s, MARGIN, Inches(3.6), Inches(6.2), Pt(2), fill=MIDGRAY)
add_textbox(s, MARGIN, Inches(3.8), Inches(6.2), Inches(3.0),
            bullets([
                ("阶段 11 一次只实际执行一条完整运输链（串行）", 13),
                ("平均服务约 70s，而 medium 任务到达间隔 30~50s", 13),
                ("服务能力 < 到达速度 → 等待任务逐渐积累", 13),
                ("任务时长含送达后离开作业区进入泊位的时间", 13),
            ], size=13, gap=9))

# 结论框
con = add_rect(s, Inches(7.1), Inches(3.6), Inches(5.6), Inches(2.4), fill=LIGHT, line=YELLOW)
rect_text(con, [
    {'runs': [("结论", 15, True, YELLOW)]},
    {'runs': [("• 不能靠缩短任务发布器间隔改善等待；", 13, False, DARK)]},
    {'runs': [("• 应由调度策略优化机器人与楼梯选择；", 13, False, DARK)]},
    {'runs': [("• 单任务策略稳定后，评估有限并发 / 泊位异步化 / 服务链拆分。", 13, False, DARK)]},
], anchor=MSO_ANCHOR.MIDDLE)

# 到达 vs 服务 示意
add_textbox(s, MARGIN, Inches(6.35), SLIDE_W - 2 * MARGIN, Inches(0.5),
            [{'runs': [("示意：任务到达速度（30~50s）快于串行服务速度（≈70s）→ 队列积累",
                        13, False, GRAY)]}])
add_footer(s, 10)

# ============================================================
# 第11页：为什么需要强化学习
# ============================================================
s = new_slide()
add_title(s, "为什么需要强化学习", "楼梯选择存在真实的全局权衡，规则难以全局优化", "11")

# 对比两种楼梯
add_textbox(s, MARGIN, Inches(1.45), Inches(8), Inches(0.4),
            [{'runs': [("跨层任务示例：同一任务，两种楼梯选择", 14, True, BLACK)]}])
cols = [
    ("方案 A · 取货侧近的楼梯", [
        "取货 Carter 到交接点距离短",
        "但目标楼层 Carter 到接收点的运输距离更长",
        "总完成时间不一定最短",
    ], MIDGRAY),
    ("方案 B · 目标侧近的楼梯", [
        "目标侧 Carter 距离更短",
        "但取货侧 / Go2 空驶成本更高",
        "取决于全局状态与未来负载",
    ], MIDGRAY),
]
cw = (SLIDE_W - 2 * MARGIN - Inches(0.3)) / 2
for i, (t, items, _) in enumerate(cols):
    x = MARGIN + i * (cw + Inches(0.3))
    h = add_rect(s, x, Inches(1.95), cw, Inches(0.5), fill=BLACK)
    rect_text(h, [{'runs': [(t, 13, True, WHITE)]}])
    add_textbox(s, x + Inches(0.1), Inches(2.6), cw - Inches(0.2), Inches(1.6),
                bullets(items, size=12.5, gap=7))

add_textbox(s, MARGIN, Inches(4.25), Inches(10), Inches(0.4),
            [{'runs': [("总成本 = 多层 Carter 距离 + Go2 空驶 + 楼梯占用 / 等待 + 接货运输 + 未来位置价值",
                        14, True, BLACK)]}])
add_textbox(s, MARGIN, Inches(4.7), SLIDE_W - 2 * MARGIN, Inches(1.0),
            bullets([
                "取货 Carter 到货物 / 发送交接点的距离",
                "Go2 当前点到楼梯入口的空驶距离",
                "楼梯通过成本、占用状态与可能等待",
                "目标楼层 Carter 到接收点、接货后到目标平台的距离",
                "各机器人近期任务负载与未来位置价值",
            ], size=13, gap=6))

con = add_rect(s, MARGIN, Inches(6.0), SLIDE_W - 2 * MARGIN, Inches(0.95), fill=LIGHT, line=BORDER)
rect_text(con, [{'runs': [("结论：", 14, True, BLUE),
                          ("规则调度器可用，但难以针对长期等待、负载均衡与未来任务进行全局优化 —— 这正是阶段 12 / 强化学习要解决的问题，不能再用「象限对应固定楼梯」的规则。",
                           14, False, DARK)]}], anchor=MSO_ANCHOR.MIDDLE)
add_footer(s, 11)

# ============================================================
# 第12页：阶段12工作与验收标准
# ============================================================
s = new_slide()
add_title(s, "阶段 12 · 强化学习接口准备", "把阶段 11 的可变结构状态与组合动作，转换为稳定、可学习、可测试的策略接口", "12")
add_status_tag(s, SLIDE_W - MARGIN - Inches(3.3), Inches(0.5), "接口准备，非直接训练 RL", BLUE, w=Inches(3.3), size=11)

col_l = [
    ("固定维度观测编码", [
        "任务槽位 K=8，不足补零并带 task_mask",
        "固定机器人排列顺序，编码类型 / 楼层 / 位置 / 载货 / 故障 / 负载",
        "编码 4 楼梯占用、两侧位置与候选代价",
        "距离、时间、累计量统一归一化，同状态重复编码逐元素一致",
    ]),
    ("组合动作候选编码", [
        "任务 × 取货Carter × Go2 × 楼梯 × 发送点 × 接收点 × 接货Carter × 任务后策略",
        "策略只在合法候选中选择；跨层任务保留多个楼梯选项",
        "显式提供取货侧 / Go2 / 跨楼层 / 接货侧 / 目标平台成本",
    ]),
    ("动作掩码压力测试", [
        "忙碌 / 故障 / 带货 / 已有任务的机器人不能接单",
        "Carter 不能上下楼，同层任务不能调用 Go2",
        "楼梯 / 交接点 / 区域占用时屏蔽冲突动作；无合法任务进入受控 WAIT",
    ]),
]
col_r = [
    ("统一接口", [
        "规则调度器与学习策略读取同一份观测 / 候选 / 掩码",
        "调用同一个环境 step()",
        "不替换 SCAN-Planner，不训练低层运动控制",
    ]),
    ("阶段 12 验收标准（8 条）", [
        "① 20 任务张量形状固定不变",
        "② 相同种子 / 状态编码完全一致",
        "③ 四类任务均至少生成一个合法动作",
        "④ 每个跨层任务至少保留两个不同楼梯的合法候选",
        "⑤ 异常状态非法动作屏蔽率 100%",
        "⑥ 动作可无歧义解码回原始协作组合",
        "⑦ 规则策略经新接口再完成 medium 20/20（不破坏封板）",
        "⑧ 输出张量规格说明、测试报告、可直接用于 RL 的样本数据",
    ]),
]

def render_col(s, x, y, w, blocks):
    yy = y
    for heading, items in blocks:
        add_textbox(s, x, yy, w, Inches(0.4),
                    [{'runs': [(heading, 14, True, BLACK)]}])
        add_rect(s, x, yy + Inches(0.4), w, Pt(1.8), fill=MIDGRAY)
        add_textbox(s, x + Inches(0.08), yy + Inches(0.55), w - Inches(0.16), Inches(0.05), [{'runs': [("", 1)]}])
        # 用紧凑段落
        paras = []
        for it in items:
            paras.append({'runs': [("▪ ", 12, True, GRAY), (it, 12, False, DARK)], 'space_after': 4, 'line_spacing': 1.12})
        add_textbox(s, x + Inches(0.08), yy + Inches(0.5), w - Inches(0.16), Inches(3.6), paras)
        yy += Inches(2.35)

cw = (SLIDE_W - 2 * MARGIN - Inches(0.5)) / 2
render_col(s, MARGIN, Inches(1.6), cw, col_l)
render_col(s, MARGIN + cw + Inches(0.5), Inches(1.6), cw, col_r)
add_footer(s, 12)

# ============================================================
# 第13页：风险与边界
# ============================================================
s = new_slide()
add_title(s, "风险与验证边界", "当前通过的是逻辑规划、状态机与 RViz 可视化验收，不是完整物理仿真最终通过", "13")

risks = [
    ("当前为 RViz / 逻辑验收", "规划与协作逻辑在 RViz 验证；Gazebo 动力学步态、抖动、碰撞与最终视觉效果留待项目收尾统一优化。", YELLOW),
    ("并发资源竞争未验证", "阶段 11 串行执行；两条运输链同时竞争同一楼梯 / 交接点 / 区域时的排队、公平性与死锁恢复属后续扩展。", YELLOW),
    ("Gazebo 动力学与泊位物理可达性", "泊位逻辑层唯一且不重叠；最终需复核模型占用、外形尺寸与 SCAN-Planner 末端可达性。", YELLOW),
    ("随机扰动暂时关闭", "当前优先使用既定确定性路线，不加入随机扰动，避免干扰验收对比。", GRAY),
]
rh = Inches(1.25)
for i, (t, d, c) in enumerate(risks):
    y = Inches(1.55) + i * (rh + Inches(0.16))
    box = add_rect(s, MARGIN, y, SLIDE_W - 2 * MARGIN, rh, fill=WHITE, line=BORDER)
    add_rect(s, MARGIN, y, Inches(0.1), rh, fill=c)
    add_textbox(s, MARGIN + Inches(0.3), y + Inches(0.12), Inches(4.6), Inches(1.0),
                [{'runs': [(t, 14, True, BLACK)]}])
    add_textbox(s, MARGIN + Inches(5.1), y + Inches(0.12), SLIDE_W - MARGIN - Inches(5.4), Inches(1.0),
                [{'runs': [(d, 12.5, False, DARK)], 'line_spacing': 1.15}])

con = add_rect(s, MARGIN, Inches(7.0) - Inches(0.7), SLIDE_W - 2 * MARGIN, Inches(0.6), fill=LIGHT, line=BORDER)
rect_text(con, [{'runs': [("边界表述：本阶段通过 = 逻辑规划 + 状态机 + RViz 可视化验收。", 13, True, YELLOW)]}])
add_footer(s, 13)

# ============================================================
# 第14页：后续路线图
# ============================================================
s = new_slide()
add_title(s, "后续路线图", "阶段 12 → 阶段 13 → 训练与泛化 → Gazebo 收尾", "14")

milestones = [
    ("阶段 12 · 接口", "张量与动作接口", [
        "固定维度观测 / 组合动作 / 动作掩码",
        "统一接口，验收 8 条标准",
    ], BLUE, "下一步"),
    ("阶段 13 · 训练准备", "奖励 / 终止条件 / 训练规则", [
        "设计奖励与终止条件",
        "定义训练规则与基线",
    ], BLUE, "下一步"),
    ("训练与验证", "规则基线对比 · RL 训练 · 泛化", [
        "规则基线对比",
        "强化学习训练",
        "泛化 / 压力测试",
    ], BLUE, "下一步"),
    ("项目收尾", "Gazebo 统一优化", [
        "动力学 / 碰撞 / 步态",
        "泊位物理可达性复核",
        "最终演示效果",
    ], BLUE, "下一步"),
]
mw = (SLIDE_W - 2 * MARGIN - 3 * Inches(0.3)) / 4
my = Inches(1.6)
mh = Inches(3.1)
for i, (phase, sub, items, color, tag) in enumerate(milestones):
    x = MARGIN + i * (mw + Inches(0.3))
    box = add_rect(s, x, my, mw, mh, fill=WHITE, line=BLACK, line_w=1.3)
    hd = add_rect(s, x, my, mw, Inches(0.72), fill=BLACK)
    rect_text(hd, [{'runs': [(phase, 13, True, WHITE)]}, {'runs': [(sub, 10.5, False, MIDGRAY)]}])
    add_textbox(s, x + Inches(0.14), my + Inches(0.85), mw - Inches(0.28), Inches(2.0),
                bullets(items, size=11.5, gap=7))
    add_status_tag(s, x + Inches(0.14), my + mh - Inches(0.48), tag, color, w=Inches(1.1), h=Inches(0.32), size=10)
    if i < 3:
        add_rect(s, x + mw + Inches(0.02), my + mh / 2 - Inches(0.2), Inches(0.26), Inches(0.4),
                 fill=BORDER, shape=MSO_SHAPE.RIGHT_ARROW)

# 下一阶段验收目标（最后一页必须给出明确目标）
goal = add_rect(s, MARGIN, Inches(5.15), SLIDE_W - 2 * MARGIN, Inches(1.5), fill=LIGHT, line=BLUE)
rect_text(goal, [
    {'runs': [("下一阶段验收目标", 15, True, BLUE)]},
    {'runs': [("完成阶段 12 张量接口并满足 8 条验收标准：形状稳定、编码一致、四类任务合法动作、跨层 ≥2 楼梯候选、",
               13, False, DARK)]},
    {'runs': [("非法动作屏蔽 100%、动作可逆解码、规则策略 medium 20/20 回归通过、输出可直接用于 RL 的样本数据。",
               13, False, DARK)]},
], anchor=MSO_ANCHOR.MIDDLE)
add_footer(s, 14)

# ---------- 保存 ----------
out = "/Users/lab4099/Desktop/Mujoco/work/0806进展与问题汇报.pptx"
prs.save(out)
print("已生成：", out)
print("页数：", len(prs.slides.__iter__.__self__._sldIdLst))
