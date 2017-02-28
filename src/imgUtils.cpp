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

cv::Mat imTranslate(cv::Mat image, cv::Point offset)
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
    if (readMode == cv::IMREAD_COLOR || img.channels() > 1)
        cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::resize(img, img, imSize, 0, 0, cv::INTER_CUBIC);
    if (useHistogramEqualization)
        cv::equalizeHist(img, img);
    return imSplitPatches(img, patchCounts);
}
