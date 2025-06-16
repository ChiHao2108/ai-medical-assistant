from flask import Flask, request, jsonify, render_template 
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure Google Generative AI with your API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# Dùng mô hình miễn phí, phản hồi nhanh
model = genai.GenerativeModel('gemini-1.5-flash-latest')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask_text', methods=['POST'])
def ask_text():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "Không có câu hỏi được cung cấp"}), 400

    try:
        medical_prompt = (
            "Bạn là một trợ lý y tế AI. Dựa trên triệu chứng sau, hãy gợi ý một số khả năng bệnh nhẹ thường gặp, và các thuốc không kê đơn (OTC) phổ biến có thể giúp giảm nhẹ triệu chứng. "
            "Chỉ nêu các thuốc đã được chứng minh an toàn, dễ mua tại nhà thuốc (ví dụ: paracetamol, oresol, thuốc ho thảo dược,...) "
            "**KHÔNG KÊ ĐƠN CHÍNH THỨC. KHÔNG ĐƯỢC NÊU THUỐC KHÁNG SINH, CORTICOID, HOẶC CÁC THUỐC CẦN TOA.** "
            "Nếu không chắc, hãy gợi ý người dùng đến bác sĩ hoặc dược sĩ để xác nhận. "
            "Triệu chứng người dùng: " + user_question +
            "\n\nBắt đầu câu trả lời bằng: 'Lưu ý: Các thông tin sau chỉ mang tính tham khảo.'"
        )

        response = model.generate_content(
            medical_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300  # Giới hạn độ dài trả lời
            )
        )
        return jsonify({"answer": response.text})
    except Exception as e:
        print(f"Lỗi khi hỏi AI: {e}")
        return jsonify({"error": f"Có lỗi xảy ra: {str(e)}"}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "Không có hình ảnh được cung cấp"}), 400

    image_file = request.files['image']
    text_input = request.form.get('text_input', '')

    if image_file.filename == '':
        return jsonify({"error": "Không có file hình ảnh nào được chọn"}), 400

    try:
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if text_input:
            medical_image_prompt = (
                "Bạn là một trợ lý y tế AI. Dựa trên triệu chứng sau, hãy gợi ý một số khả năng bệnh nhẹ thường gặp, và các thuốc không kê đơn (OTC) phổ biến có thể giúp giảm nhẹ triệu chứng. "
                "Chỉ nêu các thuốc đã được chứng minh an toàn, dễ mua tại nhà thuốc (ví dụ: paracetamol, oresol, thuốc ho thảo dược,...) "
                "**KHÔNG KÊ ĐƠN CHÍNH THỨC. KHÔNG ĐƯỢC NÊU THUỐC KHÁNG SINH, CORTICOID, HOẶC CÁC THUỐC CẦN TOA.** "
                "Nếu không chắc, hãy gợi ý người dùng đến bác sĩ hoặc dược sĩ để xác nhận. "
                "Triệu chứng người dùng: " + text_input +
                "\n\nBắt đầu câu trả lời bằng: 'Lưu ý: Các thông tin sau chỉ mang tính tham khảo.'"
            )
        else:
            medical_image_prompt = (
                "Bạn là một trợ lý phân tích hình ảnh y tế ảo. Vui lòng mô tả ngắn gọn các đặc điểm trực quan trong hình ảnh này liên quan đến y tế. "
                "Không phân tích dư thừa, không đưa ra chẩn đoán hay kết luận cuối cùng. "
                "Nếu có thể, hãy liệt kê dưới dạng gạch đầu dòng các yếu tố bất thường (nếu có)."
                "Luôn nhắc người dùng tham khảo bác sĩ để xác nhận. "
            )

        contents = [medical_image_prompt, image]
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300
            )
        )
        return jsonify({"analysis": response.text})
    except Exception as e:
        print(f"Lỗi khi phân tích hình ảnh: {e}")
        return jsonify({"error": f"Có lỗi xảy ra trong quá trình phân tích hình ảnh: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

