from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io
import json
from datetime import datetime

# Load .env
load_dotenv()

app = Flask(__name__)

# API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in .env")
genai.configure(api_key=GOOGLE_API_KEY)

# Dùng mô hình mới nhất, phản hồi nhanh
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Giới hạn lượt dùng miễn phí mỗi tháng
USAGE_FILE = "usage.json"
LIMIT_PER_MONTH = 100  # lượt miễn phí/tháng
WARNING_THRESHOLD = 90  # cảnh báo nếu vượt 90%

# Tạo usage file nếu chưa tồn tại
def init_usage():
    if not os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, 'w') as f:
            json.dump({}, f)

def get_usage_data():
    with open(USAGE_FILE, 'r') as f:
        return json.load(f)

def save_usage_data(data):
    with open(USAGE_FILE, 'w') as f:
        json.dump(data, f)

def check_usage_status():
    now = datetime.now()
    key = f"{now.year}-{now.month}"
    data = get_usage_data()
    used = data.get(key, 0)
    percent_used = int((used / LIMIT_PER_MONTH) * 100)
    if used >= LIMIT_PER_MONTH:
        return {"blocked": True, "message": "Đã hết lượt miễn phí trong tháng này."}
    elif percent_used >= WARNING_THRESHOLD:
        return {"blocked": False, "message": f"⚠️ Bạn đã dùng {used}/{LIMIT_PER_MONTH} lượt trong tháng này."}
    else:
        return {"blocked": False, "message": None}

def increase_usage():
    now = datetime.now()
    key = f"{now.year}-{now.month}"
    data = get_usage_data()
    data[key] = data.get(key, 0) + 1
    save_usage_data(data)

init_usage()

@app.route('/')
def index():
    usage_status = check_usage_status()
    return render_template('index.html', usage_status=usage_status)

@app.route('/ask_text', methods=['POST'])
def ask_text():
    usage_status = check_usage_status()
    if usage_status['blocked']:
        return jsonify({"error": usage_status['message']}), 403

    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Vui lòng nhập mô tả triệu chứng"}), 400

    try:
        prompt = (
            "Bạn là một trợ lý y tế AI. Dựa trên mô tả triệu chứng sau, hãy: "
            "- Phân tích ngắn gọn những bệnh nhẹ thường gặp. "
            "- Gợi ý thuốc không kê đơn (OTC) phù hợp như Paracetamol, thuốc ho thảo dược, v.v. "
            "- Nếu triệu chứng không rõ, hãy khuyên người dùng đi khám. "
            "- QUAN TRỌNG: Gợi ý TÊN CHUYÊN KHOA phù hợp và TÊN MỘT SỐ BỆNH VIỆN phổ biến tại Việt Nam có chuyên khoa đó.\n\n"
            "Mô tả: " + question +
            "\n\nLưu ý: Chỉ mang tính chất tham khảo. Không được chẩn đoán thay bác sĩ."
        )

        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=400))
        increase_usage()
        return jsonify({"answer": response.text, "warning": usage_status['message']})
    except Exception as e:
        print("Lỗi ask_text:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    usage_status = check_usage_status()
    if usage_status['blocked']:
        return jsonify({"error": usage_status['message']}), 403

    files = request.files.getlist('image')
    text_input = request.form.get('text_input', '')

    if not files or len(files) == 0:
        return jsonify({"error": "Chưa gửi hình ảnh nào"}), 400

    try:
        image_list = []
        for f in files:
            if f and f.filename:
                img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes))
                image_list.append(image)

        if len(image_list) == 0:
            return jsonify({"error": "Tập tin ảnh không hợp lệ"}), 400

        if text_input:
            prompt = (
                "Bạn là một trợ lý y tế AI. Hãy xem hình ảnh (có thể là da, tổn thương ngoài da,...) kèm mô tả sau để phân tích sơ bộ. "
                "Gợi ý vài khả năng bệnh nhẹ nếu có dấu hiệu rõ ràng, kèm cách xử lý cơ bản tại nhà và thuốc không kê đơn (nếu phù hợp). "
                "Tuyệt đối không đề xuất thuốc cần kê đơn. Sau cùng, nếu cần, hãy gợi ý chuyên khoa hoặc bệnh viện phù hợp tại Việt Nam. "
                f"Mô tả: {text_input} \n\nLưu ý: Các thông tin sau chỉ mang tính tham khảo."
            )
        else:
            prompt = (
                "Bạn là trợ lý AI y tế. Hãy xem hình ảnh sau và mô tả các đặc điểm y tế trực quan nếu có. "
                "Không chẩn đoán, chỉ nêu ra điểm bất thường (nếu có). Gợi ý người dùng nên đi khám nếu cần."
                "\n\nLưu ý: Các thông tin sau chỉ mang tính tham khảo."
            )

        contents = [prompt]
        for img in image_list:
            contents.append(img)

        response = model.generate_content(contents, generation_config=genai.types.GenerationConfig(max_output_tokens=400))
        increase_usage()
        return jsonify({"analysis": response.text, "warning": usage_status['message']})
    except Exception as e:
        print("Lỗi analyze_image:", e)
        return jsonify({"error": f"Lỗi xử lý ảnh: {str(e)}"}), 500

@app.route('/usage_check')
def usage_check():
    status = check_usage_status()
    return jsonify({
        "warning": status['message'] if status['message'] else None,
        "blocked": status['blocked']
    })

if __name__ == '__main__':
    app.run(debug=True)

