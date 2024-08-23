#
# This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
# Last modified by David J Turner (turne540@msu.edu) 23/08/2024, 12:21. Copyright (c) The Contributors
#

coverage run -m unittest discover
coverage report
coverage xml
coverage html

genbadge coverage


