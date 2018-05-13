#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import requests
import numpy as np
from HTMLParser import HTMLParser

class AbbreviationResolver(HTMLParser):

    team_abbreviation = None
    team_fullname = None
    _team_min_length = 4

    def __init__(self):
        HTMLParser.__init__(self)
        self.HTML_data = []

    def handle_data(self, data):
        self.HTML_data.append(data)

    @staticmethod
    def query_team_data():

        """
        Queries full name corresponding to the given abbrevation in 'team_abbreviation' in football related topics
        using https://abbreviations.com
        :return: -
        """

        # Send HTTP request with the given abbreviation
        abbrev = AbbreviationResolver.team_abbreviation
        http_req = requests.get("https://www.abbreviations.com/serp.php?st={}&p=99999".format(abbrev))

        # Find the appropriate part in the HTML reply
        html_reply = http_req.content
        description_reg = re.compile('<p class="desc">.*p>')
        desc_res = description_reg.search(html_reply)
        html_subs = html_reply[desc_res.start():desc_res.end()]

        # Instantiate parser and feed with HTML subset
        parser = AbbreviationResolver()
        parser.feed(html_subs)

        # Load possible teams from file
        with open('football_teams.txt', 'r') as f:
            teams_fromfile = np.array(f.read().splitlines())
        f.close()

        # Find intersection between HTML Data and fromfile teams
        team_name_intersect = np.intersect1d(np.array(parser.HTML_data), teams_fromfile)
        team_fullname = np.array(filter(lambda x: len(x) > AbbreviationResolver._team_min_length, team_name_intersect))

        AbbreviationResolver.team_fullname = team_fullname

