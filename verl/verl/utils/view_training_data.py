#!/usr/bin/env python3
"""
用于启动本地服务器查看训练数据分析页面的脚本
"""

import os
import sys
import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path
from flask import Flask, jsonify

def list_experiments(base_log_dir):
    """
    列出所有可用的实验
    
    Args:
        base_log_dir: 基础日志目录
    
    Returns:
        实验名称列表
    """
    base_log_dir = Path(base_log_dir)
    if not base_log_dir.exists():
        print(f"错误: 日志目录 '{base_log_dir}' 不存在")
        sys.exit(1)
    
    experiments = [d.name for d in base_log_dir.iterdir() if d.is_dir()]
    if not experiments:
        print(f"警告: 在 '{base_log_dir}' 中没有找到任何实验")
        return []
    
    return sorted(experiments)

def select_experiment(experiments):
    """
    让用户选择要查看的实验
    
    Args:
        experiments: 实验名称列表
    
    Returns:
        选择的实验名称
    """
    if not experiments:
        print("没有可用的实验")
        sys.exit(1)
    
    print("可用的实验:")
    for i, exp in enumerate(experiments):
        print(f"{i+1}. {exp}")
    
    while True:
        try:
            choice = input("请选择实验编号 [1-{}] (或 'q' 退出): ".format(len(experiments)))
            if choice.lower() == 'q':
                sys.exit(0)
                
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                return experiments[idx]
            else:
                print(f"无效的选择。请输入 1 到 {len(experiments)} 之间的数字。")
        except ValueError:
            print("请输入有效的数字。")

def run_server(data_path, experiment_name, static_dir, port=8000):
    """
    启动本地HTTP服务器来查看训练数据分析页面
    
    Args:
        data_path: 数据日志目录的路径
        experiment_name: 实验名称
        static_dir: 静态网页文件目录
        port: 服务器端口号
    """
    # 验证目录存在
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"错误: 数据目录 '{data_path}' 不存在")
        sys.exit(1)
    
    static_dir = Path(static_dir)
    if not static_dir.exists():
        print(f"错误: 网页文件目录 '{static_dir}' 不存在")
        sys.exit(1)
    
    # 切换到项目根目录，以便相对路径正确解析
    script_dir = Path(__file__).parent.parent.parent
    os.chdir(script_dir)
    
    # 数据路径映射，记录用于debug
    print(f"数据路径: {data_path}")
    print(f"静态目录: {static_dir}")
    print(f"工作目录: {os.getcwd()}")
    
    # 设置简单的HTTP服务器
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            """重写路径转换方法，处理绝对路径数据请求"""
            import urllib.parse
            
            # 解析URL获取查询参数
            parsed_url = urllib.parse.urlparse(self.path)
            path_part = parsed_url.path  # 仅URL路径部分，不含查询参数
            query_params = dict(urllib.parse.parse_qsl(parsed_url.query))
            
            # 调试信息
            print(f"\n[调试] 转换路径: {self.path}")
            print(f"[调试] 路径部分: {path_part}")
            print(f"[调试] 查询参数: {query_params}")
            
            # 首先执行普通路径转换 - 但如果处理特殊路径，我们会完全替换它
            translated_path = super().translate_path(path)
            
            # 检查静态资源路径 - 只关注路径部分，忽略查询参数
            if path_part.startswith('/scripts/analysis/'):
                # 处理静态资源文件请求
                resource_path = path_part.replace('/scripts/analysis/', '')
                if not resource_path or resource_path == '/' or resource_path == 'index.html':
                    return str(static_dir / 'index.html')
                return str(static_dir / resource_path)
            
            # 检查是否是API调用或数据文件请求
            if path_part.startswith('/api/') or path_part.startswith('/data/'):
                # 如果是API调用或直接数据文件请求
                if 'data_path' in query_params:
                    # 使用查询参数中的数据路径
                    requested_data_path = urllib.parse.unquote(query_params['data_path'])
                    print(f"[调试] 从查询参数获取数据路径: {requested_data_path}")
                    
                    # 检查是否是以/fs开头的特殊绝对路径
                    is_fs_path = requested_data_path.startswith('/fs/')
                    
                    # 如果是API调用，可能需要在data_path基础上追加相对路径
                    if path_part.startswith('/api/'):
                        api_path = path_part.replace('/api/', '')
                        if requested_data_path.startswith('/'):
                            # 绝对路径 - 直接使用，不要添加工作目录前缀
                            final_path = str(Path(requested_data_path) / api_path)
                            print(f"[调试] API请求最终路径(绝对): {final_path}")
                            return final_path
                        else:
                            # 相对路径
                            final_path = os.path.join(os.getcwd(), requested_data_path, api_path)
                            print(f"[调试] API请求最终路径(相对): {final_path}")
                            return final_path
                    
                    # 直接数据文件请求
                    if requested_data_path.startswith('/'):
                        # 绝对路径 - 直接使用，不要添加任何前缀
                        print(f"[调试] 数据请求最终路径(绝对): {requested_data_path}")
                        return requested_data_path
                    else:
                        # 相对路径
                        final_path = os.path.join(os.getcwd(), requested_data_path)
                        print(f"[调试] 数据请求最终路径(相对): {final_path}")
                        return final_path
                elif path_part.startswith('/data/'):
                    # 直接访问/data/下的文件，使用实验数据路径
                    rel_path = path_part.split('/data/', 1)[1]
                    final_path = str(data_path / rel_path)
                    print(f"[调试] /data/请求最终路径: {final_path}")
                    # 检查是否是以/fs开头的特殊绝对路径
                    if str(data_path).startswith('/fs/'):
                        return final_path
                    return final_path
            
            # 如果没有匹配特殊路径，使用默认的路径转换
            # 但首先检查是否是尝试访问以/fs开头的路径
            if '/fs/' in translated_path:
                # 这可能是一个被错误转换的/fs/路径
                # 提取可能的真实路径
                parts = translated_path.split('/fs/')
                if len(parts) > 1:
                    corrected_path = '/fs/' + parts[1]
                    print(f"[调试] 修正错误的路径转换: {translated_path} -> {corrected_path}")
                    return corrected_path
            
            print(f"[调试] 默认路径转换结果: {translated_path}")
            return translated_path
            
        def do_GET(self):
            import urllib.parse
            
            # 解析请求
            parsed_url = urllib.parse.urlparse(self.path)
            path_part = parsed_url.path
            query_params = dict(urllib.parse.parse_qsl(parsed_url.query))
            
            print(f"\n[调试] 收到GET请求: {self.path}")
            print(f"[调试] 请求头: {self.headers}")
            
            # 处理根目录访问，重定向到分析页面
            if self.path == '/':
                print("[调试] 处理根路径请求，重定向到分析页面")
                self.send_response(302)
                # 传递编码后的data_path作为参数
                encoded_path = urllib.parse.quote(str(data_path))
                redirect_url = f'/scripts/analysis/?data_path={encoded_path}&experiment={experiment_name}'
                self.send_header('Location', redirect_url)
                self.end_headers()
                print(f"[调试] 重定向到: {redirect_url}")
                return
            
            try:
                # 输出详细请求信息，用于调试
                print(f"[调试] 请求URL: {self.path}")
                print(f"[调试] 请求路径: {path_part}")
                if query_params:
                    print(f"[调试] 查询参数: {query_params}")
                
                # 获取实际文件路径
                file_path = self.translate_path(self.path)
                print(f"[调试] 转换后的文件路径: {file_path}")
                
                # 判断文件是否存在
                if os.path.exists(file_path) and not os.path.isdir(file_path):
                    print(f"[调试] 文件存在，处理请求: {file_path}")
                else:
                    print(f"[警告] 文件不存在: {file_path}")
                    
                # 检查是否是JSON文件请求
                if self.path.endswith('.json') or self.path.endswith('.jsonl'):
                    if os.path.exists(file_path):
                        print(f"[调试] 提供JSON文件: {file_path}")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            print(f"[调试] JSON内容前100个字符: {content[:100]}")
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        with open(file_path, 'rb') as f:
                            self.wfile.write(f.read())
                        return
                    else:
                        print(f"[错误] JSON文件不存在: {file_path}")
                    
                # 处理正常请求
                http.server.SimpleHTTPRequestHandler.do_GET(self)
            except FileNotFoundError as e:
                print(f"[错误] 文件未找到: {e}")
                self.send_error(404, f"文件未找到: {str(e)}")
            except PermissionError as e:
                print(f"[错误] 权限错误: {e}")
                self.send_error(403, f"权限错误: {str(e)}")
            except Exception as e:
                # 如果出现异常，返回错误
                print(f"[错误] 处理请求时出错: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.send_error(500, f"内部服务器错误: {str(e)}")
    
    print(f"启动服务器在端口 {port}...")
    
    try:
        with socketserver.TCPServer(("", port), CustomHandler) as httpd:
            url = f"http://localhost:{port}/"
            print(f"在浏览器中打开：{url}")
            
            # 自动在浏览器中打开
            webbrowser.open(url)
            
            print("按 Ctrl+C 停止服务器")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"错误: 端口 {port} 已被占用，请尝试其他端口")
        else:
            print(f"错误: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="启动本地服务器查看训练数据分析页面")
    parser.add_argument("log_dir", help="数据日志目录路径")
    parser.add_argument("--experiment", help="要查看的特定实验名称")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口号 (默认: 8000)")
    parser.add_argument("--list", action="store_true", help="列出所有可用的实验")
    parser.add_argument("--static-dir", default="scripts/analysis", help="静态网页文件目录 (默认: scripts/analysis)")
    
    args = parser.parse_args()
    
    base_log_dir = args.log_dir
    port = args.port
    static_dir = args.static_dir
    
    # 列出所有实验
    experiments = list_experiments(base_log_dir)
    
    if args.list:
        if experiments:
            print("可用的实验:")
            for exp in experiments:
                print(f"  - {exp}")
        sys.exit(0)
    
    # 确定要查看的实验
    if args.experiment:
        experiment = args.experiment
        if experiment not in experiments:
            print(f"错误: 实验 '{experiment}' 不存在")
            sys.exit(1)
    elif len(experiments) == 1:
        experiment = experiments[0]
        print(f"只有一个可用的实验: {experiment}")
    else:
        experiment = select_experiment(experiments)
    
    # 启动查看器
    data_path = os.path.join(base_log_dir, experiment)
    print(f"查看实验: {experiment}")
    run_server(data_path, experiment, static_dir, port)

app = Flask(__name__)

# 设置PROJECT_ROOT路径
PROJECT_ROOT = "/fs/fast/u2020201469/models/saves/Qwen2.5-Coder-1.5B-Instruct/grpo/v1_qwen2.5-coder-7b-18k/verl_grpo/"

@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    try:
        # 获取PROJECT_ROOT目录下的所有文件夹名
        experiments = [name for name in os.listdir(PROJECT_ROOT) if os.path.isdir(os.path.join(PROJECT_ROOT, name))]
        print("experiments:", experiments)
        return jsonify(experiments)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    main() 