"""
Configuration loading and template substitution helpers.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
"""
import json
import logging
import os


def get_data_settings(file_name):
    """Load a JSON settings file and return its contents as a dict.

    Raises IOError if the file does not exist.
    """
    if os.path.exists(file_name):
        with open(file_name) as file_handle:
            data_settings = json.load(file_handle)
    else:
        logging.error(' ===> Error in reading settings file "' + file_name + '"')
        raise IOError('File not found')
    return data_settings


def substitute_keywords(template, **kwargs):
    """Replace ``{key}`` placeholders in ``template`` with the given keyword values."""
    for key, value in kwargs.items():
        template = template.replace('{' + key + '}', str(value))
    return template
