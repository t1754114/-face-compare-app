import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(page_title="顔比較アプリ", layout="centered")
st.title("顔比較画像生成アプリ")

# --- 明度自動補正関数 ---
def auto_brightness_balance(image_pil, target_mean=150):
    image_np = np.array(image_pil)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    beta = target_mean - mean_brightness
    adjusted = cv2.convertScaleAbs(image_np, alpha=1.0, beta=beta)
    return Image.fromarray(adjusted)

# --- 顔検出関数（OpenCV使用） ---
def extract_face_region(image_pil):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_np = np.array(image_pil)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    h_img, w_img = gray.shape
    center = np.array([w_img / 2, h_img / 2])
    distances = [np.linalg.norm(center - np.array([x + fw/2, y + fh/2])) for (x, y, fw, fh) in faces]
    x, y, fw, fh = faces[np.argmin(distances)]

    # トリミング範囲の拡張（上下左右）
    y_top = max(0, int(y - fh * 0.2))
    y_bottom = min(h_img, int(y + fh * 1.5))
    x_left = max(0, int(x - fw * 0.1))
    x_right = min(w_img, int(x + fw * 1.1))

    cropped = image_pil.crop((x_left, y_top, x_right, y_bottom))
    return auto_brightness_balance(cropped)

# --- ファイルアップロード ---
st.subheader("1枚目（Before画像）をアップロード")
file1 = st.file_uploader("画像1を選択", type=["jpg", "jpeg", "png"], key="before")
date1 = st.text_input("Before画像の日付（例：05/16、省略可）")

st.subheader("2枚目（After画像）をアップロード")
file2 = st.file_uploader("画像2を選択", type=["jpg", "jpeg", "png"], key="after")
date2 = st.text_input("After画像の日付（例：05/30、省略可）")

# --- 実行ボタン ---
if st.button("比較画像を生成"):
    if file1 is None or file2 is None:
        st.error("両方の画像をアップロードしてください。")
    else:
        try:
            img1 = Image.open(file1).convert('RGB')
            img2 = Image.open(file2).convert('RGB')

            face1 = extract_face_region(img1)
            face2 = extract_face_region(img2)

            if face1 is None or face2 is None:
                st.error("どちらかの画像から顔を検出できませんでした。")
            else:
                # 高さを合わせる
                h1, h2 = face1.height, face2.height
                target_h = max(h1, h2)

                def resize(img):
                    w, h = img.size
                    scale = target_h / h
                    return img.resize((int(w * scale), target_h))

                face1 = resize(face1)
                face2 = resize(face2)

                total_w = face1.width + face2.width
                result_h = target_h + 120
                result = Image.new('RGB', (total_w, result_h), (0, 0, 0))
                result.paste(face1, (0, 0))
                result.paste(face2, (face1.width, 0))

                # ラベル描画
                draw = ImageDraw.Draw(result)
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=36)
                except:
                    font = ImageFont.load_default()

                label1 = f"Before ({date1})" if date1 else "Before"
                label2 = f"After ({date2})" if date2 else "After"

                def draw_label(label, x_offset, width):
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x = x_offset + (width - tw) // 2
                    y = target_h + 10
                    draw.rectangle([x - 10, y - 10, x + tw + 10, y + th + 10], fill=(50, 50, 100))
                    draw.text((x, y), label, fill=(255, 255, 255), font=font)

                draw_label(label1, 0, face1.width)
                draw_label(label2, face1.width, face2.width)

                st.image(result, caption="比較画像", use_column_width=True)

                # ダウンロード用
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                st.download_button("画像をダウンロード", data=buf.getvalue(), file_name="result.png", mime="image/png")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
