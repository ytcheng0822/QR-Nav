import imageio
import cv2
import numpy as np
import os

def combine_videos(fps_path, metric_path, output_path):
    # 檢查檔案是否存在
    if not os.path.exists(fps_path):
        print(f"❌ 找不到檔案: {fps_path}")
        return
    if not os.path.exists(metric_path):
        print(f"❌ 找不到檔案: {metric_path}")
        return

    print(f"⏳ 正在讀取影片: {fps_path} 與 {metric_path}...")

    # 建立影片讀取器
    fps_reader = imageio.get_reader(fps_path)
    metric_reader = imageio.get_reader(metric_path)

    # 取得原始影片的幀率 (FPS)，如果讀不到預設用 4
    fps_meta = fps_reader.get_meta_data()
    video_fps = fps_meta.get('fps', 4)

    # 建立影片寫入器
    writer = imageio.get_writer(output_path, fps=video_fps)

    frame_count = 0
    try:
        # 使用 zip 同時讀取兩個影片的影格
        for fps_frame, metric_frame in zip(fps_reader, metric_reader):
            h_img, w_img = fps_frame.shape[:2]
            h_top, w_top = metric_frame.shape[:2]

            # 自動縮放左邊的視角畫面，使其高度與右邊的地圖一致
            if h_img != h_top:
                new_w_img = int(w_img * (h_top / h_img))
                fps_frame_resized = cv2.resize(fps_frame, (new_w_img, h_top))
            else:
                fps_frame_resized = fps_frame

            # 水平拼接 (axis=1)
            combined_frame = np.concatenate((fps_frame_resized, metric_frame), axis=1)

            # 寫入合成後的畫面
            writer.append_data(combined_frame)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"  已處理 {frame_count} 幀...")

    except Exception as e:
        print(f"❌ 合成過程中發生錯誤: {e}")
    finally:
        # 確保所有檔案資源都有被正確關閉
        fps_reader.close()
        metric_reader.close()
        writer.close()

    print(f"🎉 合成完成！影片已儲存至: {output_path} (共 {frame_count} 幀)")

if __name__ == "__main__":
    # ==========================================
    # 在這裡設定你的輸入與輸出檔案路徑
    # ==========================================
    # 假設影片放在 ./tmp/trajectory_0/ 目錄下
    input_fps = "./tmp/trajectory_0/fps.mp4"
    input_metric = "./tmp/trajectory_0/metric.mp4"
    output_result = "./tmp/trajectory_0/result.mp4"

    combine_videos(input_fps, input_metric, output_result)