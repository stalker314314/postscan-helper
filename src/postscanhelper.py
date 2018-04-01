# -*- coding: utf-8 -*-

import argparse
import logging
import logging.handlers
import operator
import os

import cv2
import numpy as np


def setup_logger(logging_level=logging.INFO):
    """
    Simple logger used throughout whole code - logs both to file and console
    """
    logger = logging.getLogger('postscan-helper')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.handlers.TimedRotatingFileHandler(filename='postscan-helper.log', when='midnight', interval=1, encoding='utf-8')
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


logger = setup_logger(logging_level=logging.INFO)


def show_image(im):
    ratio = float(im.shape[1]) / float(im.shape[0])
    if im.shape[0] > 600 or im.shape[1] > 600:
        im = cv2.resize(im, (int((600 * ratio) + 0.5), 600))
    cv2.imshow('image', im)
    while True:
        key = cv2.waitKey(100) & 255
        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            logger.info('User closed windows, quitting')
            exit(0)
        if key != 255:
            break
    return key


def warp_contour(global_context, image, c):
    # Stolen from https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

    # now that we have our screen contour, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = c.reshape(4, 2)
    rect = np.zeros((4, 2), dtype='float32')

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    min_width = min(int(width_a), int(width_b))
    min_height = min(int(height_a), int(height_b))

    if min_width < global_context.min_detection_size or min_height < global_context.min_detection_size:
        logger.debug('Contour with size %dx%d is lower than minimal image detection size of %d',
                     min_width, min_height, global_context.min_detection_size)
        return None

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [min_width - 1, 0],
        [min_width - 1, min_height - 1],
        [0, min_height - 1]], dtype='float32')

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    _M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, _M, (min_width, min_height))


def auto_rotate_image(original_image):
    ret_image = original_image
    faces_count = [0, 0, 0, 0]
    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    for i in range(4):
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
        faces_count[i] = len(faces)
        gray_img = cv2.transpose(gray_img)
        gray_img = cv2.flip(gray_img, 1)

    max_index, max_value = max(enumerate(faces_count), key=operator.itemgetter(1))
    min_value = min(faces_count)
    # Simple heuristic when we don't want to rotate image
    if min_value == max_value:
        # If all rotations have same number of faces
        return ret_image
    if max_value != 0 and faces_count[0] == max_value:
        # If starting position has same number of faces as maximum
        return ret_image
    logger.info("Max faces detected (%d) for rotation %d", max_value, max_index)
    for i in range(max_index):
        ret_image = cv2.transpose(ret_image)
        ret_image = cv2.flip(ret_image, 1)
    return ret_image


def process_contour(global_context, file_path, image, contour, idx):
    peri = cv2.arcLength(contour, True)
    approx_contour = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx_contour) != 4:
        logger.debug('After approximation, this contour is not rectangle, skipping')
        return False
    warp = warp_contour(global_context, image, approx_contour)
    if warp is None:
        logger.debug('Contour %d cannot be warped', idx)
        return False

    image_to_show =warp
    if not global_context.no_auto_rotate:
        image_to_show = auto_rotate_image(image_to_show)

    if global_context.auto_wb_fix:
        image_to_show = cv2.xphoto.createSimpleWB().balanceWhite(image_to_show)
    if global_context.borders > 0:
        current_image_width, current_image_height = image_to_show.shape[:2]
        image_to_show = image_to_show[1:current_image_width - 1, 1:current_image_height - 1]
    while True:
        if not global_context.non_interactive:
            response = show_image(image_to_show)
            logger.info('User pressed %d', response)
        else:
            response = ord('s')

        if response == 27:  # ESC clicked, skip image
            return False
        elif chr(response) == 'q':  # exit
            exit(0)
        elif chr(response) == 'i':  # ignore
            return False
        elif chr(response) == 'r':  # rotate
            image_to_show = cv2.transpose(image_to_show)
            image_to_show = cv2.flip(image_to_show, 1)
        elif chr(response) == 's':  # save
            original_filename = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = global_context.output_directory
            filename = os.path.join(output_dir, '%s-%d.jpg' % (original_filename, idx))
            cv2.imwrite(filename, image_to_show, [cv2.IMWRITE_JPEG_QUALITY, global_context.jpeg_quality])
            return True
        elif chr(response) == 'a':  # auto level
            image_to_show = cv2.xphoto.createSimpleWB().balanceWhite(image_to_show)
        elif chr(response) == 'b':  # crop image
            current_image_width, current_image_height = image_to_show.shape[:2]
            image_to_show = image_to_show[1:current_image_width-1, 1:current_image_height-1]
        elif chr(response) == 'u':  # undo all and reset
            image_to_show = warp


def process_file(global_context, file_path):
    image = cv2.imread(file_path)
    if image is None:
        # Hmm, OpenCV cannot cope with this format, let's try with imagemagick, if it is available
        try:
            from wand.image import Image
            with Image(filename=file_path) as im:
                img_buffer = np.asarray(bytearray(im.make_blob()), dtype=np.uint8)
            image = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        except ImportError:
            logger.info('Imagemagick module is not installed. Try installing it (Wand) to be able to load this image')

        if image is None:
            logger.warning('File %s cannot be loaded, skipping it', file_path)
            return

    image_width, image_height = image.shape[:2]
    if not global_context.non_interactive:
        response = show_image(image)
        if chr(response) == 'i':
            logger.info('User ignored this image')
            return

    current_threshold = global_context.threshold
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    while True:
        _, thresh = cv2.threshold(image_gray, current_threshold, 255, 0)
        if not global_context.threshold_tweak:
            break
        if global_context.non_interactive:
            break

        response = show_image(thresh)
        if chr(response) == '+':
            current_threshold = min(current_threshold + 5, 255)
            continue
        if chr(response) == '-':
            current_threshold = max(current_threshold - 5, 0)
            continue
        break

    (_, contours, hierarchies) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        logger.info('No contours detected, skipping this image')
        return
    logger.info('Found %d contours', len(contours))

    parent_hierarchy_ids = [i for i, h in enumerate(hierarchies[0]) if h[3] == -1]
    if len(parent_hierarchy_ids) > 1:
        parent_hierarchy_id = -1
        logger.info('Multiple root contours, that\'s fine, will detect only root contours for photo scans')
    else:  # Only one root contour
        parent_hierarchy_id = parent_hierarchy_ids[0]

    counter = 1
    for i, c in enumerate(contours):
        hierarchy = hierarchies[0, i]
        # Check if this is root element
        if parent_hierarchy_id != -1 and parent_hierarchy_id == i:
            # Check if scanned photo is only one (no children, contour is almost as big as image itself)
            if hierarchy[2] == -1 and cv2.contourArea(c) > 0.8 * image_width * image_height:
                if process_contour(global_context, file_path, image, c, counter):
                    counter = counter + 1
            else:
                is_not_big_enough = 'is not big as image, not confident there is scanned photo in image'
                has_children = 'it has children, maybe threshold is not good, lets continue with children'
                logger.info('Contour is parent contour, but %s',
                            (is_not_big_enough if hierarchy[2] == -1 else has_children))
                continue
        else:
            if parent_hierarchy_id == -1 and hierarchy[3] != -1:
                logger.debug('Skipping contour with idx %d - not a root contour', i)
            if parent_hierarchy_id != -1 and hierarchy[3] != parent_hierarchy_id:
                logger.debug('Skipping contour with idx %d - it is not direct child of root contour', i)
                continue
            if process_contour(global_context, file_path, image, c, counter):
                counter = counter + 1


def create_global_context():
    parser = argparse.ArgumentParser(
        description='PostScan Helper - helper utility to find, crop, rotate and fix your scanned photographs en masse')
    parser.add_argument('input_directory', metavar='input-directory',
                        help='Directory containing scanned photographs')
    parser.add_argument('-o', '--output-directory', default='.',
                        help='Directory where output files will be save. Default value is current directory.')
    parser.add_argument('-t', '--threshold', default=240,
                        help='Threshold value to detect scanned photographs on original image.'
                             'Valid only in non-interactive mode.')
    parser.add_argument('--min_detection_size', default=100,
                        help='Contours whole width or height are below this value will '
                             'not be considered as candidates for detected scanned photographs.')
    parser.add_argument('-b', '--borders', default=5,
                        help='Size of borders (in pixels) that will be cut from each side of the image.')
    parser.add_argument('--threshold-tweak', action='store_true',
                        help='Shows additional dialog where threshold can be tweaked.')
    parser.add_argument('-a', '--non-interactive', action='store_true',
                        help='Run without showing any dialog')
    parser.add_argument('--no-auto-rotate', action='store_true',
                        help='Do not try to auto-rotate image by guessing rotation from contextual information')
    parser.add_argument('--auto-wb-fix', action='store_true',
                        help='Fixes white balance when run in non-interactive mode.')
    parser.add_argument('--jpeg-quality', default=90,
                        help='JPEG quality with which to save images')
    parser.add_argument('-v', '--version', action='version', version='PostScan Helper 0.1')

    args = parser.parse_args()

    if not os.path.isdir(args.input_directory):
        error_msg = 'Input directory {0} is missing. You need to create it first'.format(args.input_directory)
        parser.error(error_msg)

    if not os.path.isdir(args.output_directory):
        error_msg = 'Output directory {0} is missing. You need to create it first'.format(args.output_directory)
        parser.error(error_msg)

    if args.threshold_tweak and args.non_interactive:
        error_msg = 'You cannot tweak threshold with --threshold-tweak while running in non-interactive mode'
        parser.error(error_msg)

    if args.borders < 0:
        error_msg = 'Value for borders must be non-negative'
        parser.error(error_msg)

    return args


def main():
    global_context = create_global_context()
    input_dir = global_context.input_directory
    for file_path in [os.path.join(input_dir, f) for f in os.listdir(input_dir)]:
        logger.info('Processing file %s', file_path)
        process_file(global_context, file_path)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
