/* --------------------
Image operations
-------------------- */
#include "imgUtils.h"

cv::Mat imReadAndDisplay(std::string imagePath, std::string windowName, cv::ImreadModes readMode)
{
    std::cout << "Reading image: " << imagePath << std::endl;
    cv::Mat img = cv::imread(imagePath, readMode);
    if (windowName != "")
    {
        cv::imshow(windowName, img);
        cv::waitKey(1); // allow window redraw
    }
    return img;
}

cv::Mat imTranslate(const cv::Mat& image, cv::Point offset)
{
    cv::Rect source = cv::Rect(cv::max(0, -offset.x), cv::max(0, -offset.y), image.cols - abs(offset.x), image.rows - abs(offset.y));
    cv::Rect target = cv::Rect(cv::max(0, offset.x), cv::max(0, offset.y), image.cols - abs(offset.x), image.rows - abs(offset.y));
    cv::Mat trans = cv::Mat::zeros(image.size(), image.type());
    image(source).copyTo(trans(target));
    return trans;
}

cv::Mat imFlip(cv::Mat image, FlipCode flipCode)
{
    cv::Mat flip;
    cv::flip(image, flip, flipCode);
    return flip;
}

std::vector<cv::Mat> imSyntheticGeneration(cv::Mat image)
{
    std::vector<cv::Mat> synth(6);
    synth[0] = image;
    synth[1] = imTranslate(image, cv::Point(4, 0));
    synth[2] = imTranslate(image, cv::Point(0, 4));
    synth[3] = imTranslate(image, cv::Point(-4, 0));
    synth[4] = imTranslate(image, cv::Point(0, -4));
    synth[5] = imFlip(image, HORIZONTAL);
    return synth;
}

std::vector<cv::Mat> imSyntheticGenerationScaleAndTranslation(const cv::Mat image, int nScales, int translationSize, double minScale)
{
    double scaleJumps = (1 - minScale) / nScales;
    std::vector<cv::Mat> synthImages;
    synthImages.push_back(image);
    int initSize = image.rows; 
    cv::Size dummySize(0, 0);
    std::cout << "scaleJumps: " << scaleJumps << " minScale: " << minScale << std::endl; 

    for(double scale = 1; scale > minScale; scale -= scaleJumps){
        cv::Mat resizedImage;
        int newSize = (int)initSize*scale;
        cv::Rect newRect(0, 0, newSize, newSize);
        int totalPixelDifference = initSize - newSize;
        int startingPoint = (totalPixelDifference % 2) / 2;
        std::cout << "initSize: " << initSize << " newsize: " << newSize << std::endl; 
        std::cout << "totalPixelDifference: " << totalPixelDifference << " translationSize: " << translationSize << std::endl; 
        if (translationSize < totalPixelDifference) {
            cv::Mat cropedImage;
            for( int x = startingPoint; x < totalPixelDifference; x += translationSize){
                for( int y = startingPoint; y < totalPixelDifference; y += translationSize){
                    newRect.x = x;
                    newRect.y = y;
                    cv::Mat image_roi = image(newRect);
                    image_roi.copyTo(cropedImage);
                    synthImages.push_back(cropedImage);
                    // std::stringstream ss;
                    // ss << "cropedImage_" << scale << "_" << x << "_" << y <<  ".jpg";
                    // cv::imwrite(ss.str(), image_roi);
                }
            }
        }
    }

    return synthImages;
}



std::vector<cv::Mat> imSplitPatches(cv::Mat image, cv::Size patchCounts)
{
    if (patchCounts == cv::Size(0, 0) || patchCounts == cv::Size(1, 1))
    {
        std::vector<cv::Mat> vImg(1);
        vImg[0] = image;
        return vImg;
    }
    else if (image.size().width % patchCounts.width == 0 && image.size().height % patchCounts.height == 0)
    {
        // Define and return image patches
        cv::Size patchSize(image.size().width / patchCounts.width, image.size().height / patchCounts.height);
        std::vector<cv::Mat> patches(patchCounts.width * patchCounts.height);
        int i = 0;
        for (int r = 0; r < image.rows; r += patchSize.height)
            for (int c = 0; c < image.cols; c += patchSize.width)
                patches[i++] = image(cv::Range(r, r + patchSize.height), cv::Range(c, c + patchSize.width));
        return patches;
    }
    return std::vector<cv::Mat>();
}

std::vector<cv::Mat> imPreprocess(std::string imagePath, cv::Size imSize, cv::Size patchCounts, bool useHistogramEqualization,
                                  std::string windowName, cv::ImreadModes readMode)
{
    cv::Mat img = imReadAndDisplay(imagePath, windowName, readMode);
    return imPreprocess(img, imSize, patchCounts, useHistogramEqualization, windowName, readMode);
}

std::vector<cv::Mat> imPreprocess(cv::Mat img, cv::Size imSize, cv::Size patchCounts, bool useHistogramEqualization,
                                  std::string windowName, cv::ImreadModes readMode)
{
    std::cout << "Preprocessing image: " << img.rows << " cols: " << img.cols << std::endl;
    if (windowName != "")
    {
        cv::imshow(windowName, img);
        cv::waitKey(1); // allow window redraw
    }
    if (readMode == cv::IMREAD_COLOR || img.channels() > 1)
        cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::resize(img, img, imSize, 0, 0, cv::INTER_CUBIC);
    if (useHistogramEqualization)
        cv::equalizeHist(img, img);
    return imSplitPatches(img, patchCounts);
}
