import os                                                                                   # Duyệt thư mục train, test
import numpy as np                                                                          # Lưu trữ dữ liệu ảnh, tính ma trận
import seaborn as sns                                                                       # Vẽ ma trận nhầm lẫn
import matplotlib.pyplot as plt                                                             # Hiển thị kết quả ma trận
from skimage.io import imread                                                               # Đọc ảnh từ file
from skimage.color import rgb2gray                                                          # Giảm 3 kênh -> 1 kênh
from skimage.transform import resize                                                        # Đưa tất cả ảnh về cùng kích thước
from skimage.feature import hog     
from sklearn.decomposition import PCA                                                       # Nén dữ liệu ảnh
from sklearn.naive_bayes import GaussianNB                                                  # Mô hình GaussianNB trong Naive Bayes
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix         # Tính accuracy, f1-score, precision, confusion matrix

# === Đường dẫn dữ liệu ===
train_dir = r'C:\Users\Nguyen Thanh Nam\Downloads\Do_an\archive\brisc2025\classification_task\train'
test_dir  = r'C:\Users\Nguyen Thanh Nam\Downloads\Do_an\archive\brisc2025\classification_task\test'

# === Hàm đọc dữ liệu + HOG ===
def load_dataset_with_hog(folder_path, img_size=(64, 64)):
    X, y = [], []                                                                           # X chứa dữ liệu ảnh, y chứa nhãn
    classes = sorted(os.listdir(folder_path))                                               # danh sách tên lớp
    for label, class_name in enumerate(classes):                                            # Duyệt các thư mục con
        class_dir = os.path.join(folder_path, class_name)                                  
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):                                                  # Duyệt từng ảnh trong thư mục
            img_path = os.path.join(class_dir, file)
            try:
                img = imread(img_path)                                                      # Đọc ảnh

                if img.ndim == 3 and img.shape[2] == 3:                                     # Chuyển sang xám nếu ảnh có 3 kênh
                    img = rgb2gray(img)

                img_resized = resize(img, img_size)                                         # Resize ảnh

                # Trích xuất đặc trưng HOG
                hog_features = hog(
                    img_resized,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    transform_sqrt=True
                )

                X.append(hog_features)
                y.append(label)
            except Exception as e:
                print(f"Lỗi đọc {img_path}: {e}")
    return np.array(X), np.array(y), classes                                                # Trả về mảng ảnh, vector nhãn, các lớp (thư mục con)

# === Đọc dữ liệu ===
X_train, y_train, classes = load_dataset_with_hog(train_dir)
X_test, y_test, _ = load_dataset_with_hog(test_dir)

print("Train:", X_train.shape, "Test:", X_test.shape)                                       # Hiển thị kích thước dữ liệu
print("Classes:", classes)

# === Giảm chiều bằng PCA ===
pca = PCA(n_components=100, random_state=42)                                                # Giảm chiều từ 1024 -> 100
X_train_pca = pca.fit_transform(X_train)                                                    # Học từ tập train
X_test_pca = pca.transform(X_test)                                                          # Áp dụng vào tập test

print("Variance giữ lại:", np.sum(pca.explained_variance_ratio_))                           # In ra phần phương sai giữ lại (cho biết mức độ còn lại của thông tin)

# === Huấn luyện Gaussian Naive Bayes ===
clf = GaussianNB()                                                                          # Tạo mô hình
clf.fit(X_train_pca, y_train)                                                               # Train mô hình

# === Dự đoán ===
y_pred = clf.predict(X_test_pca)                                                            # Áp dụng mô hình

# === Kết quả ===
acc = accuracy_score(y_test, y_pred)                                                        
print(f"\n🎯 Độ chính xác trên tập test: {acc:.4f}\n")

print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3], target_names=classes, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Ma trận nhầm lẫn:\n", cm)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn (HOG + PCA + Naive Bayes)")
plt.tight_layout()
plt.show()

print("Các lớp xuất hiện trong y_test:", np.unique(y_test))
print("Các lớp xuất hiện trong y_pred:", np.unique(y_pred))

