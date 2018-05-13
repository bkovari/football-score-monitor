#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from matplotlib import pyplot as plt
from abbreviation import AbbreviationResolver

import football_monitor
import pytesseract
import cv2
import Tkinter as tk
import numpy as np
import logging
import time
import re
import threading

class ImageHandler(object):

    def __init__(self):
        self.scoreboard_image = None
        self.time_image = None
        self.time_text = None
        self.teams_goals_image = None
        self.teams_goals_text = None

        self.video_source_path = 'football_short.mp4'
        self.export_image_path = 'football.png'

        logging.basicConfig(level=logging.WARN)

    def extract_image_from_video(self):

        """
        Extracts image from video and saves on disk with specified period.

        :param path_to_video: Path to video and video name with file format
        :param export_image_path: Export image path and image name with file format
        :return: -
        """

        vidcap = cv2.VideoCapture(self.video_source_path)
        count = 0
        #success = True
        image_lst = []
        while True:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
            success, image = vidcap.read()
            image_lst.append(image)

            ## Stop when last frame is identified
            if count > 1:
                if np.array_equal(image, image_lst[1]):
                    break
                image_lst.pop(0)  # Clean the list
            cv2.imwrite(self.export_image_path, image)  # save frame as PNG file
            logging.info('{}.sec reading a new frame: {} '.format(count, success))
            count += 1
            eImageExported.set()
            time.sleep(1)

    def localize_scoreboard_image(self):
        """
        Finds the scoreboard table in the upper corner
        using Canny edge detection, sets scoreboard_image
        and exports the picture as 'scoreboard_table.png'

        :return: True when scoreboard is found
                 False when scoreboard is not found
        """

        # Read a snapshot image from the video and convert to gray
        snapshot_image = cv2.imread(self.export_image_path)
        grayscale_image = cv2.cvtColor(snapshot_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        canny_points = cv2.Canny(grayscale_image, 200, 200)

        # Crop upper corner of Canny image
        canny_upper_left_corner = canny_points[2:75, 0:400]

        # DEBUG
        # cv2.imshow('canny_upper_left_corner', canny_upper_left_corner)
        # cv2.waitKey(0)

        # Localize the scoreboard edges on Canny image
        idx_lst = []
        try:
            for i in range(canny_upper_left_corner.shape[0]):
                pxl_cnt = 0
                for j in range(canny_upper_left_corner.shape[1]):
                    if pxl_cnt > 100:
                        idx_lst.append(i)
                    if canny_upper_left_corner[i, j] == 255:
                        pxl_cnt += 1
            upper_row = min(idx_lst)
            lower_row = max(idx_lst)

            # Export the localized scoreboard
            self.scoreboard_image = grayscale_image[upper_row:lower_row, 0:400]
            cv2.imwrite('scoreboard_table.png', self.scoreboard_image)

            # # DEBUG
            # cv2.imshow('scoreboard_table',self.scoreboard_image)
            # cv2.waitKey(0)
            # #

            return True

        except Exception as e:
            if len(idx_lst) == 0:
                logging.info(e)
                logging.info("No scoreboard found!")
                return False

    def split_scoreboard_image(self):
        """
        Splits the scoeboard image into two parts, sets 'time_image' and 'teams_goals_image'
        and exports as 'time_table.png' and 'teams_goals_table.png'
        Left image represents the time.
        Right image represents the teams and goals.

        :return: -
        """
        self.time_image = self.scoreboard_image[:, 0:175]
        cv2.imwrite('time_table.png', self.time_image)

        self.teams_goals_image = self.scoreboard_image[:, 175:]
        cv2.imwrite('teams_goals_table.png', self.teams_goals_image)

        ## DEBUG
        # cv2.imshow('scoreboard_table_left',self.time_image)
        # cv2.imshow('scoreboard_table_right',self.teams_goals_image)
        # cv2.waitKey(0)
        ##

    def enlarge_scoreboard_images(self, enlarge_ratio):
        """
        Enlarges 'time_table.png' and 'teams_goals_table.png'

        :param enlarge_ratio: Defines the enlarging size (e.g 2-3x)
        :return: -
        """
        self.time_image = cv2.resize(self.time_image, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)
        self.teams_goals_image = cv2.resize(self.teams_goals_image, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)

        ## DEBUG
        # cv2.imshow('time_image_enlarged',self.time_image)
        # cv2.imshow('teams_goals_enlarged',self.teams_goals_image)
        # cv2.waitKey(0)
        ##


    def _get_time_from_image(self):
        """
        Preprocesses time_image transformations for OCR.
        Exports 'time_ocr_ready.png' after the manipulations.
        Reads match time from 'time_ocr_ready.png' using Tesseract.
        Applies result to time_text.

        :return: True: string is found
                 False: string is not found
        """

        # HISTOGRAM
        # plt.hist(self.time_image.ravel(), 256, [0, 256])
        # plt.title("Time OCR Image Histogram")
        # plt.show()

        # Count nonzero to determine contrast type
        ret, threshed_img = cv2.threshold(self.time_image, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        nonzero_pxls = np.count_nonzero(threshed_img)
        pxls_limit = np.size(threshed_img)/4

        # Applying Special Thresholding and Morphological Transformation for Time OCR preprocess
        if nonzero_pxls < pxls_limit:
         self.time_image = cv2.GaussianBlur(self.time_image, (3, 3), 0)

        ret, self.time_image = cv2.threshold(self.time_image, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)

        if nonzero_pxls < pxls_limit:
            self.time_image = cv2.erode(self.time_image, kernel,iterations=1)
        else:
            self.time_image = cv2.dilate(self.time_image,kernel,iterations=1)

        cv2.imwrite('time_ocr_ready.png', self.time_image)
        self.time_text = pytesseract.image_to_string(Image.open('time_ocr_ready.png'),config="-psm 6")
        logging.info('Time OCR text: {}'.format(self.time_text))


        # DEBUG
        # cv2.imshow('time_OCR_read',self.time_image)
        # cv2.waitKey(0)
        #
        if self.time_text is not None:
            return True
        return False

    def _get_teams_goals_from_image(self):
        """
        Preprocesses teams_goals_image with transformations for OCR.
        Exports 'teams_goals_ocr_ready.png' after the manipulations.
        Reads teams and goals information from 'teams_goals_ocr_ready.png' using Tesseract.
        Applies result to teams_goals_text.

        :return: True: string is found
                 False: string is not found

        """
        # HISTOGRAM
        # plt.hist(self.teams_goals_image.ravel(), 256, [0, 256])
        # plt.title("Teams goals OCR Image Histogram")
        # plt.show()

        # Applying Thresholding for Teams goals OCR preprocess
        ret, self.teams_goals_image = cv2.threshold(self.teams_goals_image, 180, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('teams_goals_ocr_ready.png', self.teams_goals_image)
        self.teams_goals_text = pytesseract.image_to_string(Image.open('teams_goals_ocr_ready.png'))
        logging.info('Teams and goals OCR text: {}'.format(self.teams_goals_text))

        ## DEBUG
        # cv2.imshow('teams_goals_OCR_read',self.teams_goals_image)
        # cv2.waitKey(0)
        ##

        if self.teams_goals_text is not None:
            return True
        return False

    def get_scoreboard_texts(self):
        """
        Returns an array of strings including OCR read time, teams and goals texts.
        :return: numpy array 'scoreboard_texts'
                 scoreboard_texts[0] : time text value
                 scoreboard_texts[1] : teams and goals text value

        """

        # Read text values using Tesseract OCR
        time_text_exists = self._get_time_from_image()
        teams_goals_text_exists = self._get_teams_goals_from_image()

        scoreboard_texts = []
        # Use values on successful read
        if time_text_exists and teams_goals_text_exists:
            scoreboard_texts.append(self.time_text)
            scoreboard_texts.append(self.teams_goals_text)
            scoreboard_texts = np.array(scoreboard_texts)

        return scoreboard_texts

    def play_match_video(self):

        cap = cv2.VideoCapture(self.video_source_path)
        count = 0

        while (cap.isOpened()):
            cap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)
            time.sleep(1)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


class Match(object):

    def __init__(self):
        self.scoreboard_text_values = None

        self.home_score = 0
        self.home_score_temp = 0

        self.home_team = None
        self.home_team_temp = 0
        self.home_team_fullname = None
        self.home_team_identified = False

        self.opponent_score = 0
        self.opponent_score_temp = 0

        self.opponent_team = None
        self.opponent_team_temp = None
        self.opponent_team_fullname = None
        self.opponent_team_identified = False

        self.match_time = None
        self.match_time_temp = None
        self._match_time_prev = []


    def analize_scoreboard(self):
        while True:
            try:
                eImageExported.wait()
                scoreboard.localize_scoreboard_image()
                scoreboard.split_scoreboard_image()
                scoreboard.enlarge_scoreboard_images(2)
                OCR_text = scoreboard.get_scoreboard_texts()
                football_match.provide_scoreboard_text_values(OCR_text)
                football_match.update_all_match_info()
                football_match.print_all_match_info()
                eImageExported.clear()
            except Exception as e:
                logging.info(e)

    def provide_scoreboard_text_values(self,scoreboard_text_values):

        self.scoreboard_text_values = scoreboard_text_values

    def cleanse_match_score(self):
        """
        Cleanse home_score_temp and opponent_score_temp values and removes
        noisy starters and enders if present

        :return: -
        """
        score_string = self.scoreboard_text_values[1].split(' ')[1]

        result = []
        for letter in score_string:
            if letter.isdigit():
                result += letter
        self.home_score_temp = result[0]
        self.opponent_score_temp = result[1]

    def cleanse_match_teams(self):
        """
        Cleanse home_team_temp and opponent_team_temp values and removes
        noisy starter or ender if present

        :return: -
        """
        self.home_team_temp = self.scoreboard_text_values[1].split(' ')[0]
        self.opponent_team_temp = self.scoreboard_text_values[1].split(' ')[2]

        # Check and remove noisy starters and enders
        if not self.home_team_temp[0].isalpha():
            self.home_team_temp = self.home_team_temp[1:4]
        elif not self.opponent_team_temp[-1].isalpha():
            self.opponent_team_temp = self.opponent_team_temp[0:3]

    def cleanse_match_time(self):
        """
        Cleanse match_time_temp, and removes noisy starter or ender if present

        :return: -
        """

        self.match_time_temp = self.scoreboard_text_values[0]

        # Check for noisy starters and ender and clean if present
        letter_ptr = 0
        if not self.match_time_temp[letter_ptr].isdigit():
            letter_ptr += 1
        if not self.match_time_temp[letter_ptr].isdigit():
            letter_ptr += 1
            self.match_time_temp = self.match_time_temp[letter_ptr:]
            logging.info("Time text noisy starter removed.")
        elif not self.match_time_temp[-1].isdigit():
            self.match_time_temp = self.match_time_temp[0:-1]
            logging.info("Time text noisy ender removed.")

    def update_match_time(self):

        """
        Validates cleansed match_time_temp with regular expression and sets match_time if valid value exists

        :return: True: time has been updated
                 False: time has not been updated
        """

        # Check if the OCR read value is valid
        time_expr = re.compile('\d\d:\d\d')
        res = time_expr.search(self.match_time_temp)

        if res is None:
            return False

        last_valid_timeval = self.match_time_temp[res.start():res.end()]
        self._match_time_prev.append(last_valid_timeval)

        # Check validity between last time values
        if last_valid_timeval < self._match_time_prev[len(self._match_time_prev)-2]:
            # Minute error occured - minute remain unchanged
            if last_valid_timeval[0:2] < self._match_time_prev[len(self._match_time_prev)-2][0:2]:
                logging.warn("Minute error occured: minute remain unchanged!")
                fixed_minutes = self._match_time_prev[len(self._match_time_prev)-2][0:2]
                last_valid_timeval = fixed_minutes + last_valid_timeval[2:]
            else:
                # Second error occured - auto increment second
                logging.warn("Second error occured: auto incremented second!")
                seconds = self._match_time_prev[len(self._match_time_prev)-2][-2:]
                fixed_seconds = str(int(seconds)+1)
                last_valid_timeval = last_valid_timeval[:-2] + fixed_seconds

        # Free unnecessary time values
        if len(self._match_time_prev) > 2:
            self._match_time_prev.pop(0)

        self.match_time = last_valid_timeval
        return True

    def update_match_score(self):
        """
        Validates cleansed score with regular expression

        :return: True: score matches the regexp
                 False: score does not match the regexp
        """
        score_expr = re.compile('\d-\d')
        res = score_expr.search(self.scoreboard_text_values[1])

        if res is None:
            return False

        self.home_score = self.home_score_temp
        self.opponent_score = self.opponent_score_temp
        return True

    def update_match_team(self,selected_team):

        """
        Sets cleansed home_team or opponent_team values if not set before

        :return: -
        """
        if selected_team == 'home':
            self.home_team = self.home_team_temp
            self.home_team_identified = True

        elif selected_team == 'opponent':
            self.opponent_team = self.opponent_team_temp
            self.opponent_team_identified = True

    def update_all_match_info(self):
        """
        Attempts to update match infos:
        time, teams, score
        :return: True: update succeed
                 False: update failed
        """
        if len(self.scoreboard_text_values[0]) > 0 and len(self.scoreboard_text_values[1]) > 0:
            try:
                # Clean OCR read time value and update time if valid
                self.cleanse_match_time()
                self.update_match_time()

                # Clean OCR read score value and update score if valid
                self.cleanse_match_score()
                self.update_match_score()


                # Clean OCR read team values and set teams if valid and necessary
                self.cleanse_match_teams()

                if self.home_team_identified is False:
                    self.update_match_team('home')

                if self.opponent_team_identified is False:
                    self.update_match_team('opponent')

            except Exception as e:
                logging.info(e)
                logging.info("Unable to update match info for some reason")
        else:
            logging.info("Unable to update match info: no text received!")

    def print_all_match_info(self):

        home_team_name = self.home_team
        opponent_team_name = self.opponent_team

        if self.home_team_fullname is not None and self.opponent_team_fullname is not None:
            home_team_name = self.home_team_fullname
            opponent_team_name = self.opponent_team_fullname

        print '{} {} {}-{} {}'.format(self.match_time,
                                      home_team_name,
                                      self.home_score,
                                      self.opponent_score,
                                      opponent_team_name)

    def resolve_team_abbreviations(self):
        """
        Resolves home_team_fullname and opponent_team_fullaname according to the identified
        abbreviations in a football match

        :return:
        """
        while self.home_team_fullname is None or self.opponent_team_fullname is None:
            if self.home_team_identified is True and self.home_team_fullname is None:
                AbbreviationResolver.team_abbreviation = self.home_team
                AbbreviationResolver.query_team_data()
                self.home_team_fullname = AbbreviationResolver.team_fullname[0]
                print(self.home_team_fullname)

            if self.opponent_team_identified is True and self.opponent_team_fullname is None:
                AbbreviationResolver.team_abbreviation = self.opponent_team
                AbbreviationResolver.query_team_data()
                self.opponent_team_fullname = AbbreviationResolver.team_fullname[0]
                print(self.opponent_team_fullname)


scoreboard = ImageHandler()
football_match = Match()

eImageExported = threading.Event()

tImageExtractor = threading.Thread(None, scoreboard.extract_image_from_video, name="ImageExtractor")
tScoreboardAnalyzer = threading.Thread(None,football_match.analize_scoreboard,name="ScoreboardAnalyzer")
tFullnameQuerier = threading.Thread(None,football_match.resolve_team_abbreviations, name="abbreviationResolver")
tVideoPlayer = threading.Thread(None,scoreboard.play_match_video,name="VideoPlayer")

tVideoPlayer.start()
tImageExtractor.start()
tScoreboardAnalyzer.start()
tFullnameQuerier.start()

monitor_ui = tk.Tk()
w,top = football_monitor.create_Football_Monitor(monitor_ui)

def user_interface_updater():

    top.lblTime.config(text=football_match.match_time)
    top.lblHomeScore.configure(text=football_match.home_score)
    top.lblOpponentScore.configure(text=football_match.opponent_score)

    if football_match.home_team_fullname is not None and football_match.opponent_team_fullname is not None:
        top.lblHomeTeam.configure(text=football_match.home_team_fullname)
        top.lblOpponentTeam.configure(text=football_match.opponent_team_fullname)
    else:
        top.lblHomeTeam.configure(text=football_match.home_team)
        top.lblOpponentTeam.configure(text=football_match.opponent_team)

    monitor_ui.after(1000, user_interface_updater)


user_interface_updater()
monitor_ui.mainloop()



