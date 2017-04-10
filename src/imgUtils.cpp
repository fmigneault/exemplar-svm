/* --------------------
Image operations
-------------------- */
#include "imgUtils.h"
#include "generic.h"

cv::Mat imReadAndDisplay(std::string imagePath, std::string windowName, cv::ImreadModes readMode)
{    
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

cv::Mat imFlip(cv::Mat image, FlipMode flipMode)
{
    cv::Mat flip;
    cv::flip(image, flip, flipMode);
    return flip;
}

cv::Mat imCrop(cv::Mat image, int x, int y, int w, int h)
{
    return image(cv::Rect(x, y, w, h));
}

cv::Mat imCrop(cv::Mat image, cv::Point p1, cv::Point p2)
{    
    return image(cv::Rect(p1, p2));
}

cv::Mat imCrop(cv::Mat image, cv::Rect r)
{
    return image(r);
}

cv::Mat imCropByRatio(cv::Mat image, double ratio, CropMode cropMode)
{
    ASSERT_LOG(ratio > 0 && ratio <= 1.0, "Ratio must be in range ]0,1]");
    int w = image.size().width;
    int h = image.size().height;
    int wc = (int)(ratio * w);
    int hc = (int)(ratio * h);

    switch (cropMode)
    {
        case TOP_LEFT:      return imCrop(image, 0, 0, wc, hc);
        case TOP_MIDDLE:    return imCrop(image, (w - wc) / 2, 0, wc, hc);
        case TOP_RIGHT:     return imCrop(image, w - wc, 0, wc, hc);
        case CENTER_LEFT:   return imCrop(image, 0, (h - hc)/2, wc, hc);
        case CENTER_MIDDLE: return imCrop(image, (w - wc) / 2, (h - hc) / 2, wc, hc);
        case CENTER_RIGHT:  return imCrop(image, w - wc, (h - hc) / 2, wc, hc);
        case BOTTOM_LEFT:   return imCrop(image, 0, h - hc, wc, hc);
        case BOTTOM_MIDDLE: return imCrop(image, (w - wc) / 2, h - hc, wc, hc);
        case BOTTOM_RIGHT:  return imCrop(image, w - wc, h - hc, wc, hc);
        default:            THROW("Invalid crop mode");
    }    
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

    for(double scale = 1; scale > minScale; scale -= scaleJumps)
    {
        cv::Mat resizedImage;
        int newSize = (int)initSize*scale;
        cv::Rect newRect(0, 0, newSize, newSize);
        int totalPixelDifference = initSize - newSize;
        int startingPoint = (totalPixelDifference % 2) / 2;

        if (translationSize < totalPixelDifference) 
        {
            cv::Mat cropedImage;
            for( int x = startingPoint; x < totalPixelDifference; x += translationSize)
            {
                for( int y = startingPoint; y < totalPixelDifference; y += translationSize)
                {
                    newRect.x = x;
                    newRect.y = y;
                    cv::Mat image_roi = image(newRect);
                    image_roi.copyTo(cropedImage);
                    synthImages.push_back(cropedImage);
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
    if (windowName != "")
    {
        cv::imshow(windowName, img);
        cv::waitKey(1); // allow window redraw
    }
    if (readMode == cv::IMREAD_COLOR || img.channels() > 1)
        cv::cvtColor(img, img, CV_BGR2GRAY);
    if (useHistogramEqualization)
        cv::equalizeHist(img, img);
    cv::resize(img, img, imSize, 0, 0, cv::INTER_CUBIC);
    return imSplitPatches(img, patchCounts);
}
