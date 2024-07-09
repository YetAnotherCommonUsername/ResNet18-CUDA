/*#include "jpg_utils.cuh"

void load_image_as_tensor(const char* filename, struct tensor* img_tensor) {
        // Load image using OpenCV
        cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
        if (img.empty()) {
            fprintf(stderr, "Error: Could not load image %s\n", filename);
            exit(1);
        }

        img_tensor->col = img.cols;
        img_tensor->row = img.rows;
        img_tensor->depth = img.channels();
        int size = img_tensor->col * img_tensor->row * img_tensor->depth;
        img_tensor->data = (float*)malloc(size * sizeof(float));

        if (img_tensor->data == NULL) {
            fprintf(stderr, "Error: Could not allocate memory for tensor data\n");
            exit(1);
        }

        // Copy image data to tensor
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                for (int c = 0; c < img.channels(); c++) {
                    img_tensor->data[(i * img.cols + j) * img.channels() + c] = img.at<cv::Vec3b>(i, j)[c];
                }
            }
        }
}
*/