#include <iostream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "zero_dce_model.hpp"

/*---------------------------------------------调用接口例子-------------------------------------------*/

ZeroDCEModel zero_model("weights/ZeroDCE_static640.onnx",true,1,true);

int main()
{

    int pipe_nums {0};
    cv::Mat image = cv::imread("media/test_data/DICM/06.jpg"); // 读取图像文件

    //------------------------------------模型预热------------------------------------
    std::cout << "Warming up the model..." << std::endl;
    cv::Mat warmup_image = cv::imread("000/media/test_data/DICM/06.jpg", cv::IMREAD_COLOR); 
    if (warmup_image.empty()) {
        std::cerr << "Error: Could not read warmup image weights11/20250424144139.009.png" << std::endl;
    } else {
        // 预热推理（执行多次以确保模型达到最佳状态）
        for (int i = 0; i < 5; i++) {
            zero_model.doInference(warmup_image);
        }
    }
    
    //------------------------------------模型预热------------------------------------
    
    
    
    //----------------判断的亮度------------------------
        cv::Mat gray;
        // 将彩色图转换为灰度图
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar mean_value = cv::mean(gray);
        std::cout << "[OpenCV] Average: " << mean_value[0] << std::endl;

        if (mean_value[0]< 235)
        {
            image = zero_model.doInference(image);
        }

    //----------------判断的亮度------------------------

    
    // 在图像中间显示 pipe_nums
    std::string text = "ZERO_DCE_MODEL" ;
    cv::Scalar color(255, 0, 255);
    cv::putText(image, text, cv::Point(110, 240), cv::FONT_HERSHEY_SIMPLEX, 1.5, color, 1);
    cv::imwrite("./121.jpg",image);
    
   
    cv::imshow("Display window", image); // 在窗口中显示图像
    cv::waitKey(0); // 等待按键事件，0表示无限等待直到有按键按下
    cv::destroyAllWindows();

    std::cout<<"qqq"<<std::endl;
    return 0;

}




// int main() {
//     // 打开视频文件或摄像头
//     cv::VideoCapture cap("media/6.mp4");
//     // 如果要打开摄像头，可以将参数改为摄像头索引（通常为 0）
//     // cv::VideoCapture cap(0);

//     //------------------------------------模型预热------------------------------------
//     std::cout << "Warming up the model..." << std::endl;
//     cv::Mat warmup_image = cv::imread("media/00001.png", cv::IMREAD_COLOR); 
//     if (warmup_image.empty()) {
//         std::cerr << "Error: Could not read warmup image weights11/20250424144139.009.png" << std::endl;
//     } else {
//         // 预热推理（执行多次以确保模型达到最佳状态）
//         for (int i = 0; i < 5; i++) {
//             model.doInference(warmup_image);
//             zero_model.doInference(warmup_image);
//         }
//     }
    
//     //------------------------------------模型预热------------------------------------

//     // 检查视频是否成功打开
//     if (!cap.isOpened()) {
//         std::cerr << "无法打开视频文件或摄像头！" << std::endl;
//         return -1;
//     }

//     // 创建一个窗口用于显示视频
//     cv::namedWindow("Video", cv::WINDOW_NORMAL);

//     cv::Mat frame;
//     while (true) {
//         // 读取一帧
//         if (!cap.read(frame)) {
//             std::cerr << "无法读取视频帧！" << std::endl;
//             break;
//         }

// //----------------判断的亮度------------------------
//         cv::Mat gray;
//         // 将彩色图转换为灰度图
//         cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
//         cv::Scalar mean_value = cv::mean(gray);
//         std::cout << "[OpenCV] Average: " << mean_value[0] << std::endl;

//         if (mean_value[0]< 150)
//         {
//             frame = zero_model.doInference(frame);
//         }

// //----------------判断的亮度------------------------
//         auto detections = model.doInference(frame);

//         model.draw(frame,detections);

//         // 显示这一帧
//         cv::imshow("Video", frame);

//         // 按下 'q' 键退出循环
//         if (cv::waitKey(30) == 'q') {
//             break;
//         }
//     }

//     // 释放资源并关闭窗口
//     cap.release();
//     cv::destroyAllWindows();

//     return 0;
// }
