import os
import subprocess
import tempfile
from flask import Flask, jsonify, send_from_directory, request

from ai.greedy import move_monsters_in_map as greedy_move
from ai.minimax import move_monsters_in_map as minimax_move
from ai.Astar import move_monsters_in_map as astar_move

app = Flask(__name__, static_folder='ui')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXE_PATH  = os.path.join(BASE_DIR, "map", "generate_map")
TXT_PATH  = os.path.join(BASE_DIR, "map", "generated_map.txt")

# Windows 本地用 .exe，Linux 不加后缀
_sa_name  = "sa.exe" if os.name == "nt" else "sa"
SA_EXE    = os.path.join(BASE_DIR, "algorithm", _sa_name)


# ===================== 页面 =====================
@app.route('/')
def index():
    return send_from_directory('ui/index', 'index.html')

@app.route('/ui/<path:filename>')
def ui_static(filename):
    return send_from_directory('ui', filename)

@app.route('/manager/<path:path>')
def manager(path):
    return send_from_directory('manager', path)


# ===================== API =====================
@app.route('/api/generate-map')
def generate_map():
    try:
        subprocess.run([EXE_PATH], check=True)

        with open(TXT_PATH, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]

        h, w = map(int, lines[0].split())
        matrix  = [list(map(int, lines[i].split())) for i in range(1, 1 + h)]
        human   = list(map(int, lines[1 + h].split()))
        monster = list(map(int, lines[2 + h].split()))

        # 补第二只怪兽
        m2_row = (monster[0] + 15) % 30
        m2_col = (monster[1] + 15) % 30
        while matrix[m2_row][m2_col] == 1:
            m2_col = (m2_col + 1) % 30

        with open(TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(f"{h} {w}\n")
            for row in matrix:
                f.write(' '.join(map(str, row)) + '\n')
            f.write(f"{human[0]} {human[1]}\n")
            f.write(f"{monster[0]} {monster[1]}\n")
            f.write(f"{m2_row} {m2_col}\n")

        response = jsonify({
            "status": "success",
            "map": matrix,
            "human": human,
            "monster": monster,
            "monster2": [m2_row, m2_col]
        })
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        return response

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ===================== SA 辅助 =====================
def _run_sa(player_pos):
    with open(TXT_PATH, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    h, w = map(int, lines[0].split())
    grid_lines = lines[1 : 1 + h]

    m_row, m_col = map(int, lines[1 + h + 1].split())  # 当前怪兽位置
    p_col, p_row = player_pos

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.txt')
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as tmp:
            for row in grid_lines:
                tmp.write(row + '\n')
            tmp.write(f"{p_row} {p_col}\n")
            tmp.write(f"{m_row} {m_col}\n")

        subprocess.run([SA_EXE, tmp_path], check=True, timeout=15)

        with open(tmp_path, 'r', encoding='utf-8') as f:
            out_lines = [l.strip() for l in f if l.strip()]

        # h+1 行之后全是路径（row col 格式）
        path_lines = out_lines[h + 1:]
        steps = [[int(x.split()[0]), int(x.split()[1])] for x in path_lines]

        # 最后一步是终点，写回主地图
        final_row, final_col = steps[-1]
        print(f"SA: monster ({m_row},{m_col}) -> ({final_row},{final_col}), steps={len(steps)}")

        with open(TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(f"{h} {w}\n")
            for gl in grid_lines:
                f.write(gl + '\n')
            f.write(f"{p_row} {p_col}\n")
            f.write(f"{final_row} {final_col}\n")

        return steps  # [[r,c], [r,c], ...] 完整路径，格式和 greedy 一致

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ===================== AI 接口 =====================
@app.route('/api/ai-step')
def ai_step():
    level = request.args.get('level', 'easy')
    hr    = int(request.args.get('hr', 0))
    hc    = int(request.args.get('hc', 0))
    player_pos = (hc, hr)   # server 内部统一 (col, row)

    try:
        if level == 'easy':
            paths = greedy_move(player_pos=player_pos)
            monster_steps = [[r, c] for (c, r) in paths[0]]

        elif level == 'normal':
            monster_steps = _run_sa(player_pos)

        elif level == 'hard':
            paths = minimax_move(player_pos=player_pos)
            print(f"minimax paths: {paths}")
            monster_steps = [[r, c] for (c, r) in paths[0][1:]]
            print(f"monster_steps sent: {monster_steps}")

        elif level == 'expert':
            p_col, p_row = player_pos

            # 读当前地图
            with open(TXT_PATH, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            h, w = map(int, lines[0].split())
            grid_lines = lines[1: 1 + h]

            paths = astar_move(player_pos=(p_row, p_col))
            path = paths[0]
            steps = path[1:3] if len(path) >= 3 else path[1:]
            monster_steps = [[r, c] for (r, c) in steps]

            # 把怪兽最终位置写回文件
            if monster_steps:
                final_row, final_col = monster_steps[-1]
                with open(TXT_PATH, 'w', encoding='utf-8') as f:
                    f.write(f"{h} {w}\n")
                    for gl in grid_lines:
                        f.write(gl + '\n')
                    f.write(f"{p_row} {p_col}\n")
                    f.write(f"{final_row} {final_col}\n")

            print(f"astar expert monster_steps: {monster_steps}")

        elif level == 'hell':
            p_col, p_row = player_pos
            paths = astar_move(player_pos=(p_row, p_col))
            print(f"hell paths raw: {paths}") 
            path0 = paths[0]
            steps0 = path0[1:3] if len(path0) >= 3 else path0[1:]
            path1 = paths[1] if len(paths) >= 2 else []
            steps1 = path1[1:3] if len(path1) >= 3 else path1[1:]
            monster_steps = [
                [[r, c] for (r, c) in steps0],
                [[r, c] for (r, c) in steps1],
            ]
            print(f"astar hell monster_steps: {monster_steps}")

        else:
            paths = greedy_move(player_pos=player_pos)
            monster_steps = [[r, c] for (c, r) in paths[0]]

        return jsonify({"status": "success", "monsters": monster_steps})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running: http://127.0.0.1:{port}/")
    app.run(debug=False, host='0.0.0.0', port=port)