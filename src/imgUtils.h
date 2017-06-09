/* --------------------
Image operations
-------------------- */
#ifndef IMG_UTILS_H
#define IMG_UTILS_H

#include "generic.h"
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"
namespace bfs = boost::filesystem;

// Reads an image from file and displays it in a small window at the same time
inline cv::Mat imReadAndDisplay(std::string imagePath, std::string windowName = "", cv::ImreadModes readMode = cv::IMREAD_GRAYSCALE)
{
    cv::Mat img = cv::imread(imagePath, readMode);
    if (windowName != "")
    {
        cv::imshow(windowName, img);
        cv::waitKey(1); // allow window redraw
    }
    return img;
}

// Converts the specified image to desired image extension format (if `path` is an image file)
// Converts all found images in a directory with specified extension from one format to another (if `path` is a directory, `fromExtension` required)
// Optionally outputs the converted image(s) to the `toDirectory`
inline bool imConvert(std::string path, std::string toExtension, std::string fromExtension = "", std::string toDirectory = "")
{
    // file path specified
    if (bfs::is_regular_file(path))
    {
        std::string writePath;
        if (toDirectory != "") {
            bfs::path tmpPath = toDirectory;
            bfs::path imgPath = path;
            tmpPath = tmpPath / imgPath.stem();
            writePath = tmpPath.string() + toExtension;
        }
        else {
            bfs::path imgPath = path;
            imgPath = imgPath.parent_path() / imgPath.stem();
            writePath = imgPath.string() + toExtension;
        }

        bfs::path parentDir = writePath;
        parentDir = parentDir.parent_path();
        bfs::create_directories(parentDir);

        cv::UMat img = cv::imread(path).getUMat(cv::ACCESS_READ);
        return cv::imwrite(writePath, img);
    }
    // directory path specified
    else if (bfs::is_directory(path))
    {
        bfs::directory_iterator endDir;
        for (bfs::directory_iterator it(path); it != endDir; ++it) {
            std::string subPath = it->path().string();
            // sub-path is also a directory
            if (bfs::is_directory(subPath) && subPath != toDirectory) {
                bfs::path subOutDir = bfs::path(toDirectory) / it->path().filename();
                imConvert(subPath, toExtension, fromExtension, subOutDir.string());
            }
            // sub-path is a file of specified extension
            else if (it->path().extension() == fromExtension)
                if (!imConvert(subPath, toExtension, fromExtension, toDirectory)) return false;
        }
        return true;
    }

    return false;
}

// Translation of an image with XY pixel offset
template<typename TMat>
inline TMat imTranslate(const TMat& image, cv::Point offset)
{
    cv::Rect source = cv::Rect(cv::max(0, -offset.x), cv::max(0, -offset.y), image.cols - abs(offset.x), image.rows - abs(offset.y));
    cv::Rect target = cv::Rect(cv::max(0, offset.x), cv::max(0, offset.y), image.cols - abs(offset.x), image.rows - abs(offset.y));
    TMat trans = TMat::zeros(image.size(), image.type());
    image(source).copyTo(trans(target));
    return trans;
}

// Flip modes for 'imFlip'
enum FlipMode { VERTICAL = 0, HORIZONTAL = 1, BOTH = -1, NONE = -2 };
// Flips an image in horizontal/vertical/both directions
template<typename TMat>
inline TMat imFlip(const TMat& image, FlipMode flipMode)
{
    TMat flip;
    if (flipMode == FlipMode::NONE) {
        image.copyTo(flip);
        return flip;
    }

    cv::flip(image, flip, flipMode);
    return flip;
}

// Resizes an image to the specified dimensions
template<typename TMat>
inline TMat imResize(const TMat& image, cv::Size size, cv::InterpolationFlags interpolMethod = cv::INTER_AREA)
{
    TMat resized;
    cv::resize(image, resized, size, 0, 0, interpolMethod);
    return resized;
}

// Crops an image with specified inputs
template<typename TMat>
inline TMat imCrop(const TMat& image, int x, int y, int w, int h)
{
    return imCrop(image, cv::Rect(x, y, w, h));
}

// Crops an image with specified inputs
template<typename TMat>
inline TMat imCrop(const TMat& image, cv::Point p1, cv::Point p2)
{
    return imCrop(image, cv::Rect(p1, p2));
}

// Crops an image with specified inputs
template<typename TMat>
inline TMat imCrop(const TMat& image, cv::Rect r)
{
    TMat crop;
    image(r).copyTo(crop);
    return crop;
}

// Crop modes for 'imCropByRatio'
enum CropMode { TOP_LEFT, TOP_MIDDLE, TOP_RIGHT, CENTER_LEFT, CENTER_MIDDLE, CENTER_RIGHT, BOTTOM_LEFT, BOTTOM_MIDDLE, BOTTOM_RIGHT };
// Crops an image so that the resulting image corresponds to the specified ratio and method
template<typename TMat>
inline TMat imCropByRatio(const TMat& image, double ratio, CropMode cropMode = CENTER_MIDDLE)
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
    case CENTER_LEFT:   return imCrop(image, 0, (h - hc) / 2, wc, hc);
    case CENTER_MIDDLE: return imCrop(image, (w - wc) / 2, (h - hc) / 2, wc, hc);
    case CENTER_RIGHT:  return imCrop(image, w - wc, (h - hc) / 2, wc, hc);
    case BOTTOM_LEFT:   return imCrop(image, 0, h - hc, wc, hc);
    case BOTTOM_MIDDLE: return imCrop(image, (w - wc) / 2, h - hc, wc, hc);
    case BOTTOM_RIGHT:  return imCrop(image, w - wc, h - hc, wc, hc);
    default:            THROW("Invalid crop mode");
    }
}

// Returns a vector containing the original image followed by multiple synthetic images generated from the original
template<typename TMat>
inline std::vector<TMat> imSyntheticGeneration(const TMat& image, size_t translationOffset = 4, double scale = 0.5, size_t minSize = 20)
{
    ASSERT_LOG(translationOffset > 0, "Translation offset must be greater than zero");
    ASSERT_LOG(scale > 0 && scale < 1.0, "Scale must be in range ]0,1[");
    ASSERT_LOG(minSize > 0, "Scale minimum size must be greater than zero");

    std::vector<TMat> synth(6);
    image.copyTo(synth[0]);
    synth[1] = imTranslate(image, cv::Point(translationOffset, 0));
    synth[2] = imTranslate(image, cv::Point(0, translationOffset));
    synth[3] = imTranslate(image, cv::Point(-translationOffset, 0));
    synth[4] = imTranslate(image, cv::Point(0, -translationOffset));
    synth[5] = imFlip(image, HORIZONTAL);
    
    size_t size = std::floor(image.rows() * (1 - scale));
    while (size > minSize) {
        synth.push_back(imResize(image), cv::Size(size, size));
        size -= std::floor(image.rows() * (1 - scale));
    }

    return synth;
}

// Returns a vector containing the patches generated by splitting the image with specified parameters
// Optionally returns only references to new area (all patches share the same memory)
// By default, patches memory is copied to avoid invalid data accesses and non-continuous memory
template<typename TMat>
inline std::vector<TMat> imSplitPatches(const TMat& image, cv::Size patchCounts, bool noCopy = false)
{
    if (patchCounts == cv::Size(0, 0) || patchCounts == cv::Size(1, 1))
    {
        std::vector<TMat> vImg(1);
        if (noCopy)
            vImg[0] = image;        
        else
            image.copyTo(vImg[0]);
        return vImg;
    }
    else if (image.size().width % patchCounts.width == 0 && image.size().height % patchCounts.height == 0)
    {
        // Define and return image patches
        cv::Size patchSize(image.size().width / patchCounts.width, image.size().height / patchCounts.height);
        std::vector<TMat> patches(patchCounts.width * patchCounts.height);
        size_t i = 0;
        /* NB
            Because the patches ROI are only a sub-area of the full Mat (whole image in memory), pixels accessed by index via
            the 'Mat::data' pointer still refer to the whole image, even if passing the pointer of the created patch ROI variable.

            Because various methods access the pixel data by index offset from such pointer, the data pointer of this image as input causes 
            improperly/unexpected indexes calculation as the whole image data is employed instead of only the patch's data.

            We resolve the issue by forcing a 'copyTo' of the patch data, creating a new and distinct array containing only the patch that can be
            accessed by index in memory as expected.
        */
        for (int r = 0; r < image.rows; r += patchSize.height) {
            for (int c = 0; c < image.cols; c += patchSize.width) {
                if (noCopy)
                    patches[i++] = image(cv::Range(r, r + patchSize.height), cv::Range(c, c + patchSize.width));
                else
                    image(cv::Range(r, r + patchSize.height), cv::Range(c, c + patchSize.width)).copyTo(patches[i++]);
            }
        }
        return patches;
    }
    return std::vector<TMat>();
}

// Returns a vector of images combining patches splitting and other preprocessing steps (resizing, grayscale, hist.equal., etc.) 
template<typename TMat>
inline std::vector<TMat> imPreprocess(std::string imagePath, cv::Size imSize, cv::Size patchCounts, bool useHistEqual = false,
                                      std::string windowName = "", cv::ImreadModes readMode = cv::IMREAD_GRAYSCALE,
                                      cv::InterpolationFlags resizeMode = cv::INTER_AREA)
{
    TMat img = imReadAndDisplay(imagePath, windowName, readMode);
    return imPreprocess(img, imSize, patchCounts, useHistEqual, "", readMode, resizeMode);  // avoid displaying again
}

// Returns a vector of images combining patches splitting and other preprocessing steps (resizing, grayscale, hist.equal., etc.) 
template<typename TMat>
inline std::vector<TMat> imPreprocess(const TMat& roi, cv::Size imSize, cv::Size patchCounts, bool useHistEqual = false,
                                      std::string windowName = "", cv::ImreadModes readMode = cv::IMREAD_GRAYSCALE, 
                                      cv::InterpolationFlags resizeMode = cv::INTER_AREA)
{
    if (windowName != "")
    {
        cv::imshow(windowName, roi);
        cv::waitKey(1); // allow window redraw
    }    
    TMat img;
    roi.copyTo(img);
    if (readMode == cv::IMREAD_COLOR || img.channels() > 1)
        cv::cvtColor(img, img, CV_BGR2GRAY);
    if (useHistEqual)
        cv::equalizeHist(img, img);
    cv::resize(img, img, imSize, 0, 0, resizeMode);
    return imSplitPatches(img, patchCounts);
}

#endif/*IMG_UTILS_H*/
