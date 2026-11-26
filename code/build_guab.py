"""Build GUAB city boundaries from GADM 4.1. 根据 GADM 4.1 构建 GUAB 城市边界。

Run / 运行
    python code/build_guab.py --gadm gadm_410-levels.gpkg --output GUAB_rebuilt.gpkg

The country defaults and city exceptions below define the study units. Join
datasets by gid. The GeoPackage fid is a storage row number.
下列国家默认层级和指定城市例外共同定义统计单元。数据关联使用 gid，fid 仅为文件内部行号。
"""
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pyogrio
import shapely


# Study registry / 城市单元选择规则。Keys are GADM 4.1 administrative levels.
# All values are geographic identifiers or administrative labels.
DEFAULT_COUNTRIES = {0: ('Antarctica', 'Aruba', 'Bouvet Island', 'British Indian Ocean Territory', 'Christmas Island',
     'Clipperton Island', 'Cocos Islands', 'Curaçao', 'Falkland Islands',
     'French Southern Territories', 'Gibraltar', 'Heard Island and McDonald Island', 'Kiribati',
     'Maldives', 'Monaco', 'Niue', 'Norfolk Island', 'Paracel Islands', 'Pitcairn Islands',
     'Saint-Martin', 'Singapore', 'Sint Maarten', 'South Georgia and the South Sand',
     'Spratly Islands', 'Vatican City'),
 1: ('Akrotiri and Dhekelia', 'Andorra', 'Anguilla', 'Antigua and Barbuda', 'Armenia', 'Bahamas',
     'Bahrain', 'Barbados', 'Belize', 'Bermuda', 'Bonaire, Sint Eustatius and Saba',
     'British Virgin Islands', 'Cabo Verde', 'Cayman Islands', 'Comoros', 'Cook Islands', 'Cyprus',
     'Dominica', 'Egypt', 'French Polynesia', 'Greenland', 'Grenada', 'Guam', 'Guernsey',
     'Isle of Man', 'Israel', 'Jamaica', 'Jersey', 'Kuwait', 'Lesotho', 'Libya', 'Liechtenstein',
     'Marshall Islands', 'Mauritius', 'Mayotte', 'Moldova', 'Montenegro', 'Montserrat', 'Nauru',
     'North Macedonia', 'Northern Cyprus', 'Northern Mariana Islands', 'Palau', 'Puerto Rico',
     'Qatar', 'Romania', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Pierre and Miquelon',
     'Saint Vincent and the Grenadines', 'San Marino', 'Seychelles', 'Svalbard and Jan Mayen',
     'Timor-Leste', 'Tokelau', 'Trinidad and Tobago', 'Turks and Caicos Islands', 'Tuvalu',
     'United States Minor Outlying Isl', 'Western Sahara'),
 2: ('Afghanistan', 'Algeria', 'Angola', 'Argentina', 'Australia', 'Austria', 'Azerbaijan',
     'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burundi',
     'Cambodia', 'Canada', 'Central African Republic', 'China', 'Colombia', 'Costa Rica', 'Croatia',
     'Cuba', 'Czechia', 'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
     'Dominican Republic', 'Ecuador', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
     'Faroe Islands', 'Fiji', 'Finland', 'French Guiana', 'Gabon', 'Gambia', 'Georgia', 'Germany',
     'Ghana', 'Guadeloupe', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Honduras', 'Hungary',
     'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Italy', 'Japan', 'Jordan',
     'Kazakhstan', 'Kenya', 'Kosovo', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Liberia',
     'Lithuania', 'Malawi', 'Malaysia', 'Malta', 'Martinique', 'Mauritania', 'Micronesia',
     'Mongolia', 'Mozambique', 'Myanmar', 'México', 'Namibia', 'Netherlands', 'New Caledonia',
     'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Palestine',
     'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',
     'Republic of the Congo', 'Russia', 'Rwanda', 'Réunion', 'Saint Helena, Ascension and Tris',
     'Saint-Barthélemy', 'Samoa', 'Saudi Arabia', 'Senegal', 'Serbia', 'Sierra Leone', 'Slovakia',
     'Slovenia', 'Solomon Islands', 'Somalia', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan',
     'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria', 'São Tomé and Príncipe', 'Taiwan',
     'Tajikistan', 'Tanzania', 'Thailand', 'Tonga', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda',
     'Ukraine', 'United Arab Emirates', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu',
     'Venezuela', 'Vietnam', 'Virgin Islands, U.S.', 'Wallis and Futuna', 'Yemen', 'Zambia',
     'Zimbabwe', 'Åland'),
 3: ('Albania', 'American Samoa', 'Bangladesh', 'Bolivia', 'Bosnia and Herzegovina', 'Burkina Faso',
     'Cameroon', 'Chad', 'Chile', "Côte d'Ivoire", 'Ethiopia', 'France', 'Greece', 'Haiti',
     'Luxembourg', 'Madagascar', 'Mali', 'Morocco', 'Nepal', 'Pakistan', 'South Africa',
     'South Sudan', 'Togo', 'United Kingdom')}

# Units chosen at a different administrative level / 采用其他行政层级的单元。
EXPLICIT_UNITS = {1: ('AGO.11_1', 'ARE.3_1', 'ARE.4_1', 'ARE.5_1', 'ARE.6_1', 'BDI.2_1', 'CHL.14_1', 'CHN.HKG',
     'CHN.MAC', 'CIV.13_1', 'CUB.4_1', 'CZE.11_1', 'DOM.31_1', 'DZA.4_1', 'ECU.9_1', 'ETH.1_1',
     'ETH.5_1', 'GMB.1_1', 'GUF.1_1', 'HRV.4_1', 'IDN.7_1', 'IND.1_1', 'IND.9_1', 'IRL.6_1',
     'IRN.28_1', 'IRN.6_1', 'JPN.41_1', 'KEN.28_1', 'KEN.30_1', 'KHM.10_1', 'KHM.12_1', 'KHM.13_1',
     'KHM.16_1', 'KOR.11_1', 'KOR.16_1', 'KOR.17_1', 'KOR.1_1', 'KOR.4_1', 'KOR.5_1', 'KOR.7_1',
     'LAO.13_1', 'LKA.11_1', 'LKA.13_1', 'LKA.15_1', 'LKA.17_1', 'LKA.19_1', 'LKA.21_1', 'LKA.24_1',
     'LKA.2_1', 'LKA.4_1', 'LKA.5_1', 'LKA.6_1', 'LKA.9_1', 'LVA.3_1', 'MEX.9_1', 'MMR.15_1',
     'MOZ.5_1', 'MWI.10_1', 'MWI.12_1', 'MWI.18_1', 'MWI.19_1', 'MWI.24_1', 'NGA.15_1', 'OMN.11_1',
     'PHL.47_1', 'PRK.11_1', 'PRK.12_1', 'PRK.6_1', 'RUS.43_1', 'RWA.5_1', 'SDN.7_1', 'SLB.4_1',
     'SRB.3_1', 'SUR.7_1', 'SVK.2_1', 'SVK.3_1', 'THA.3_1', 'TKM.2_1', 'TUN.10_1', 'TUN.11_1',
     'TUN.12_1', 'TUN.13_1', 'TUN.14_1', 'TUN.15_1', 'TUN.16_1', 'TUN.17_1', 'TUN.18_1', 'TUN.19_1',
     'TUN.1_1', 'TUN.20_1', 'TUN.21_1', 'TUN.22_1', 'TUN.23_1', 'TUN.24_1', 'TUN.2_1', 'TUN.3_1',
     'TUN.4_1', 'TUN.5_1', 'TUN.6_1', 'TUN.7_1', 'TUN.8_1', 'TUN.9_1', 'TUR.11_1', 'TUR.12_1',
     'TUR.1_1', 'TUR.21_1', 'TUR.25_1', 'TUR.26_1', 'TUR.31_1', 'TUR.32_1', 'TUR.33_1', 'TUR.37_1',
     'TUR.40_1', 'TUR.41_1', 'TUR.42_1', 'TUR.47_1', 'TUR.52_1', 'TUR.53_1', 'TUR.55_1', 'TUR.56_1',
     'TUR.57_1', 'TUR.58_1', 'TUR.59_1', 'TUR.63_1', 'TUR.66_1', 'TUR.67_1', 'TUR.68_1', 'TUR.73_1',
     'TUR.75_1', 'TUR.78_1', 'TUR.7_1', 'TUR.8_1', 'TZA.2_1', 'TZA.30_1', 'UKR.11_1', 'UKR.20_1',
     'URY.10_2', 'URY.11_2', 'URY.13_2', 'URY.14_2', 'URY.15_2', 'URY.18_2', 'URY.19_2', 'URY.1_2',
     'URY.5_2', 'URY.7_2', 'VNM.12_1', 'VNM.19_1', 'VNM.23_1', 'VNM.25_1', 'VNM.27_1', 'YEM.1_1',
     'YEM.9_1', 'ZWE.2_1'),
 2: ('BGD.1.2_1', 'BGD.2.4_1', 'BGD.2.5_1', 'BGD.3.11_1', 'BGD.3.1_1', 'BGD.3.3_1', 'BGD.4.5_1',
     'BGD.5.7_1', 'BGD.6.7_1', 'BGD.7.4_1', 'BGD.8.10_1', 'BIH.2.6_1', 'BIH.3.5_1', 'FRA.8.3_1',
     'MAR.10.5_1', 'MAR.11.2_1', 'MAR.11.3_1', 'MAR.11.4_1', 'MAR.12.1_1', 'MAR.12.3_1',
     'MAR.14.4_1', 'MAR.3.2_1', 'MAR.5.1_1', 'MAR.8.5_1', 'MAR.9.5_1', 'MLI.2.3_1', 'MLI.3.3_1',
     'MLI.4.2_1', 'MLI.6.6_1', 'MLI.7.6_1', 'MLI.8.5_1', 'MLI.9.5_1', 'PAK.8.2_1', 'SSD.3.6_1',
     'ZAF.3.3_1'),
 3: ('AGO.1.2.2_1', 'AGO.14.1.1_1', 'AGO.15.9.3_1', 'AGO.2.9.4_1', 'AGO.6.6.2_1', 'AGO.8.2.4_1',
     'BEL.2.3.4_1', 'BEL.3.2.2_1', 'CAN.1.16.4_1', 'CAN.11.20.14_1', 'FIN.3.2.3_1', 'FIN.4.5.2_1',
     'FIN.5.3.4_1', 'FIN.5.6.6_1', 'ITA.16.2.15_1', 'LBR.11.1.1_1', 'LBR.5.3.11_1', 'LBR.5.3.1_1',
     'LBR.7.3.3_1', 'MOZ.9.6.1_1'),
 4: ('MDG.5.3.1.9_1', 'MDG.5.3.5.10_1', 'MDG.5.3.5.15_1')}

# Signed-off omissions from the default selection / 从默认层级排除的单元。
EXCLUDED_UNITS = {1: ('MKD.42_1', 'MKD.52_1'),
 2: ('BDI.11.3_1', 'BDI.3.5_1', 'BDI.4.4_1', 'CAN.9.21_1', 'CAN.9.22_1', 'CAN.9.23_1', 'CAN.9.24_1',
     'CHE.20.3_1', 'DEU.1.5_1', 'EST.13.12_1', 'EST.13.22_1', 'EST.9.1_1', 'GTM.19.2_1',
     'GTM.9.5_1', 'JPN.14.43_1', 'JPN.36.12_1', 'KGZ.5.6_1', 'KGZ.9.6_1', 'KHM.17.6_2',
     'KHM.2.15_2', 'KHM.21.12_2', 'KHM.4.8_2', 'KHM.6.9_2', 'MWI.13.1_1', 'MWI.13.2_1',
     'MWI.20.1_1', 'MWI.25.1_1', 'MWI.26.1_1', 'MWI.28.1_1', 'MWI.6.2_1', 'MWI.8.2_1', 'NGA.8.19_1',
     'NIC.10.1_1', 'NIC.13.1_1', 'NLD.13.1_1', 'NLD.14.1_1', 'NLD.6.1_1', 'PAN.11.5_1',
     'PAN.12.4_1', 'PAN.12.5_1', 'PAN.4.4_1', 'PER.22.6_1', 'PRY.9.18_1', 'RUS.18.44_1',
     'RUS.2.63_1', 'RUS.26.26_1', 'RUS.45.17_1', 'RUS.49.25_1', 'RUS.57.28_1', 'RUS.58.56_1',
     'RUS.62.34_1', 'RUS.68.46_1', 'RUS.72.46_1', 'RUS.75.25_1', 'RUS.77.37_1', 'RUS.78.30_1',
     'RUS.81.21_1', 'RUS.9.22_1', 'SLV.10.9_1', 'SLV.12.6_1', 'SLV.12.7_1', 'SLV.3.11_1',
     'THA.40.11_1', 'THA.64.16_1', 'TZA.1.4_1', 'TZA.1.5_1', 'TZA.11.5_1', 'TZA.12.3_1',
     'TZA.16.3_1', 'TZA.21.2_1', 'TZA.21.3_1', 'TZA.24.4_1', 'TZA.24.5_1', 'TZA.31.4_1',
     'TZA.4.6_1', 'TZA.6.6_1', 'TZA.7.1_1', 'TZA.7.2_1', 'TZA.8.8_1', 'UGA.31.1_1', 'UGA.32.1_1',
     'USA.14.51_1', 'USA.15.46_1', 'USA.23.44_1', 'USA.23.45_1', 'USA.23.46_1', 'USA.23.47_1',
     'USA.24.40_1', 'USA.33.25_1', 'USA.36.44_1', 'USA.50.34_1', 'USA.50.35_1'),
 3: ('BOL.4.10.4_2', 'BOL.4.14.2_2', 'BOL.4.15.2_2', 'BOL.4.19.5_2', 'BOL.4.5.3_2', 'FRA.1.7.3_1')}

# Source attributes used in the study registry / 登记表采用的行政属性。
UNSPECIFIED_TYPES = {'AGO.1.2.2_1', 'AGO.11_1', 'AGO.14.1.1_1', 'AGO.15.9.3_1', 'AGO.2.9.4_1', 'AGO.6.6.2_1',
 'AGO.8.2.4_1', 'ARE.3_1', 'ARE.4_1', 'ARE.5_1', 'ARE.6_1', 'ARM.4_1', 'AZE.1.2_1', 'AZE.1.3_1',
 'AZE.10.4_1', 'AZE.10.5_1', 'AZE.2.11_1', 'AZE.2.18_1', 'AZE.2.1_1', 'AZE.4.4_1', 'AZE.4.6_1',
 'AZE.6.4_1', 'AZE.7.3_1', 'AZE.9.6_1', 'BDI.2_1', 'BEL.2.3.4_1', 'BEL.3.2.2_1', 'BGD.1.2_1',
 'BGD.2.4_1', 'BGD.2.5_1', 'BGD.3.11_1', 'BGD.3.1_1', 'BGD.3.3_1', 'BGD.4.5_1', 'BGD.5.7_1',
 'BGD.6.7_1', 'BGD.7.4_1', 'BGD.8.10_1', 'BIH.2.6_1', 'BIH.3.5_1', 'CAN.1.16.4_1', 'CAN.11.20.14_1',
 'CHL.14_1', 'CHN.HKG', 'CHN.MAC', 'CIV.13_1', 'CUB.4_1', 'CZE.11_1', 'DOM.31_1', 'DZA.4_1',
 'ECU.9_1', 'ERI.1.1_1', 'ERI.2.6_1', 'ESP.7.1_1', 'ESP.7.2_1', 'ESP.8.1_1', 'ETH.1_1', 'ETH.5_1',
 'FIN.3.2.3_1', 'FIN.4.5.2_1', 'FIN.5.3.4_1', 'FIN.5.6.6_1', 'FRA.8.3_1', 'GMB.1_1', 'GUF.1_1',
 'HRV.4_1', 'HUN.1.2_1', 'HUN.1.5_1', 'HUN.10.2_1', 'HUN.11.5_1', 'HUN.12.7_1', 'HUN.13.5_1',
 'HUN.14.5_1', 'HUN.16.9_1', 'HUN.17.4_1', 'HUN.18.8_1', 'HUN.19.8_1', 'HUN.2.3_1', 'HUN.3.1_1',
 'HUN.4.8_1', 'HUN.6.2_1', 'HUN.6.6_1', 'HUN.7.10_1', 'HUN.7.4_1', 'HUN.8.2_1', 'HUN.9.3_1',
 'IDN.7_1', 'IND.1_1', 'IND.9_1', 'IRL.6_1', 'IRN.28_1', 'IRN.6_1', 'ITA.16.2.15_1', 'JAM.3_1',
 'JPN.41_1', 'KAZ.1.14_1', 'KAZ.1.3_1', 'KAZ.10.13_1', 'KAZ.10.3_1', 'KAZ.11.4_1', 'KAZ.12.10_1',
 'KAZ.12.13_1', 'KAZ.14.9_1', 'KAZ.3.2_1', 'KAZ.4.1_1', 'KAZ.5.17_1', 'KAZ.5.9_1', 'KAZ.6.1_1',
 'KAZ.8.4_1', 'KEN.28_1', 'KEN.30_1', 'KHM.10_1', 'KHM.12_1', 'KHM.13_1', 'KHM.16_1', 'KOR.11_1',
 'KOR.16_1', 'KOR.17_1', 'KOR.1_1', 'KOR.4_1', 'KOR.5_1', 'KOR.7_1', 'LAO.13_1', 'LBR.11.1.1_1',
 'LBR.5.3.11_1', 'LBR.5.3.1_1', 'LBR.7.3.3_1', 'LKA.11_1', 'LKA.13_1', 'LKA.15_1', 'LKA.17_1',
 'LKA.19_1', 'LKA.21_1', 'LKA.24_1', 'LKA.2_1', 'LKA.4_1', 'LKA.5_1', 'LKA.6_1', 'LKA.9_1',
 'LVA.3_1', 'MAR.10.5_1', 'MAR.11.2_1', 'MAR.11.3_1', 'MAR.11.4_1', 'MAR.12.1_1', 'MAR.12.3_1',
 'MAR.14.4_1', 'MAR.3.2_1', 'MAR.5.1_1', 'MAR.8.5_1', 'MAR.9.5_1', 'MDA.10_1', 'MDA.2_1', 'MDA.4_1',
 'MDG.1.4.4_1', 'MDG.2.1.4_1', 'MDG.2.1.5_1', 'MDG.3.3.4_1', 'MDG.4.2.3_1', 'MDG.5.3.1.9_1',
 'MDG.6.3.8_1', 'MEX.9_1', 'MLI.2.3_1', 'MLI.3.3_1', 'MLI.4.2_1', 'MLI.6.6_1', 'MLI.7.6_1',
 'MLI.8.5_1', 'MLI.9.5_1', 'MMR.15_1', 'MOZ.5_1', 'MOZ.9.6.1_1', 'MWI.10_1', 'MWI.12_1', 'MWI.18_1',
 'MWI.19_1', 'MWI.24_1', 'NAM.1.4_1', 'NAM.10.7_1', 'NAM.13.3_1', 'NAM.2.7_1', 'NAM.3.3_1',
 'NAM.3.4_1', 'NAM.4.9_1', 'NGA.15_1', 'OMN.11_1', 'PAK.8.2_1', 'PHL.47_1', 'PRK.11_1', 'PRK.12_1',
 'PRK.6_1', 'ROU.10_1', 'RUS.43_1', 'RWA.5_1', 'SDN.7_1', 'SLB.4_1', 'SRB.3_1', 'SSD.3.6_1',
 'SUR.7_1', 'SVK.2_1', 'SVK.3_1', 'THA.3_1', 'TKM.2_1', 'TUN.10_1', 'TUN.11_1', 'TUN.12_1',
 'TUN.13_1', 'TUN.14_1', 'TUN.15_1', 'TUN.16_1', 'TUN.17_1', 'TUN.18_1', 'TUN.19_1', 'TUN.1_1',
 'TUN.20_1', 'TUN.21_1', 'TUN.22_1', 'TUN.23_1', 'TUN.24_1', 'TUN.2_1', 'TUN.3_1', 'TUN.4_1',
 'TUN.5_1', 'TUN.6_1', 'TUN.7_1', 'TUN.8_1', 'TUN.9_1', 'TUR.11_1', 'TUR.12_1', 'TUR.1_1',
 'TUR.21_1', 'TUR.25_1', 'TUR.26_1', 'TUR.31_1', 'TUR.32_1', 'TUR.33_1', 'TUR.37_1', 'TUR.40_1',
 'TUR.41_1', 'TUR.42_1', 'TUR.47_1', 'TUR.52_1', 'TUR.53_1', 'TUR.55_1', 'TUR.56_1', 'TUR.57_1',
 'TUR.58_1', 'TUR.59_1', 'TUR.63_1', 'TUR.66_1', 'TUR.67_1', 'TUR.68_1', 'TUR.73_1', 'TUR.75_1',
 'TUR.78_1', 'TUR.7_1', 'TUR.8_1', 'TZA.2_1', 'TZA.30_1', 'UKR.11_1', 'UKR.20_1', 'URY.10_2',
 'URY.11_2', 'URY.13_2', 'URY.14_2', 'URY.15_2', 'URY.18_2', 'URY.19_2', 'URY.1_2', 'URY.5_2',
 'URY.7_2', 'VNM.12_1', 'VNM.19_1', 'VNM.23_1', 'VNM.25_1', 'VNM.27_1', 'VUT.3.4_1', 'VUT.4.13_1',
 'WLF.2.1_1', 'YEM.1_1', 'YEM.9_1', 'ZAF.3.3_1', 'ZWE.2_1'}

COUNTRY_LABEL_CHINA = {'TWN.1.1_1', 'TWN.1.2_1', 'TWN.2.1_1', 'TWN.3.1_1', 'TWN.4.1_1', 'TWN.5.1_1', 'TWN.6.1_1',
 'TWN.7.10_1', 'TWN.7.11_1', 'TWN.7.12_1', 'TWN.7.13_1', 'TWN.7.14_1', 'TWN.7.15_1', 'TWN.7.1_1',
 'TWN.7.2_1', 'TWN.7.3_1', 'TWN.7.4_1', 'TWN.7.5_1', 'TWN.7.6_1', 'TWN.7.7_1', 'TWN.7.8_1',
 'TWN.7.9_1', 'XPI', 'XSP', 'Z04.13.5_1', 'Z04.13.7_1', 'Z05.35.3_1', 'Z05.35.9_1', 'Z07.3.10_1',
 'Z07.3.12_1', 'Z07.3.13_1', 'Z07.3.15_1', 'Z07.3.16_1', 'Z07.3.17_1', 'Z07.3.18_1', 'Z07.3.1_1',
 'Z07.3.3_1', 'Z07.3.4_1', 'Z07.3.5_1', 'Z07.3.6_1', 'Z07.3.7_1', 'Z07.3.9_1', 'Z09.35.13_1'}

UNIONS = {'CAN.2.14_1': {'level': 2,
                'parts': ['CAN.2.14_1'],
                'city_level': 'ADM_2',
                'operation': 'v1_union',
                'attributes': {'city_name': 'Vancouver', 'type': '', 'engtype': ''}},
 'USA.33.3_1|USA.33.24_1|USA.33.32_1|USA.33.42_1|USA.33.44_1': {'level': 2,
                                                                'parts': ['USA.33.3_1',
                                                                          'USA.33.24_1',
                                                                          'USA.33.32_1',
                                                                          'USA.33.42_1',
                                                                          'USA.33.44_1'],
                                                                'city_level': 'ADM_2',
                                                                'operation': 'v1_union',
                                                                'attributes': {'city_name': 'New '
                                                                                            'York '
                                                                                            'City',
                                                                               'type': '',
                                                                               'engtype': ''}},
 'MDG.REPAIR.ANTANANARIVO.20261105': {'level': 3,
                                      'parts': ['MDG.1.1.5_1', 'MDG.1.1.6_1', 'MDG.1.1.7_1'],
                                      'city_level': 'ADM3_UNION',
                                      'operation': 'core_union',
                                      'attributes': {'city_name': 'Antananarivo',
                                                     'type': 'DERIVED_MULTI_UNIT_UNION',
                                                     'engtype': 'Derived administrative union'}},
 'MDG.REPAIR.TOAMASINA.20261105': {'level': 4,
                                   'parts': ['MDG.5.3.5.10_1', 'MDG.5.3.5.15_1'],
                                   'city_level': 'ADM4_UNION',
                                   'operation': 'core_union',
                                   'attributes': {'city_name': 'Toamasina',
                                                  'type': 'DERIVED_MULTI_UNIT_UNION',
                                                  'engtype': 'Derived administrative union'}}}

RELEASE = {'version': '1.0',
 'units': 48037,
 'union_wkb_sha256': {'CAN.2.14_1': '758f8e6e7ab6dc4601b82f5a4dff1bcc897cc83e484d6de7de214b305d663fc9',
                      'USA.33.3_1|USA.33.24_1|USA.33.32_1|USA.33.42_1|USA.33.44_1': 'b1647d299f2530ce48a635f57eea9cddf2cb9003d5d1491f89a56adc657a8a30',
                      'MDG.REPAIR.ANTANANARIVO.20261105': '6ef58ff103c72d17f337e757f28f719f3abc8607eca196b4482c4f4afd5d3cff',
                      'MDG.REPAIR.TOAMASINA.20261105': '2857aa0329d4b9f9dc635c749994f85d4d158f0f7b9974b529593c16bb1a4d09'},
 'unit_digest': '72cabcdaa7366666dce1563c1eee660b130dc1d6d5bf7e45beae08c891315cc9'}
BASE_FIELDS = ("city_name", "country", "default_level", "city_level", "gid", "type", "engtype")
FIELDS = (*BASE_FIELDS, "unit_level", "source_gids", "boundary_source", "boundary_status")


@dataclass
class Unit:
    fid: int
    gid: str
    default_level: str
    city_level: str
    level: int
    source_fids: tuple
    source_gids: tuple
    operation: str
    overrides: dict


def read_units(db):
    defaults = {country: level for level, names in DEFAULT_COUNTRIES.items() for country in names}
    explicit = {(level, gid) for level, names in EXPLICIT_UNITS.items() for gid in names}
    excluded = {(level, gid) for level, names in EXCLUDED_UNITS.items() for gid in names}
    candidates = {}
    blocked_parents = set()
    coarser = set()

    for level in range(5):
        hierarchy = ", ".join(f"GID_{i}" for i in range(level + 1))
        query = f"SELECT fid, COUNTRY, {hierarchy} FROM ADM_{level}"
        for source_fid, country, *ancestors in db.execute(query):
            gid = ancestors[-1]
            key = (level, gid)
            default = defaults.get(country)
            if key in explicit:
                if level < default:
                    coarser.add(key)
                elif level > default:
                    blocked_parents.add((default, ancestors[default]))
            elif level != default:
                continue
            candidates[key] = (source_fid, country, ancestors)

    composite_parts = {(u["level"], gid) for u in UNIONS.values() for gid in u["parts"]}
    units = []
    for (level, gid), (source_fid, country, ancestors) in sorted(candidates.items()):
        key = level, gid
        if key in composite_parts or key in excluded:
            continue
        if key not in explicit and (key in blocked_parents or any(
                (i, ancestor) in coarser for i, ancestor in enumerate(ancestors))):
            continue
        attrs = {"type": "", "engtype": ""} if gid in UNSPECIFIED_TYPES else {}
        if gid in COUNTRY_LABEL_CHINA:
            attrs["country"] = "China"
        units.append(Unit(len(units) + 1, gid, f"ADM_{defaults[country]}", f"ADM_{level}",
                          level, (source_fid,), (gid,), "direct", attrs))

    for gid, union in UNIONS.items():
        level = union["level"]
        parts = tuple(union["parts"])
        source_rows = [candidates[level, part] for part in parts]
        country = source_rows[0][1]
        units.append(Unit(len(units) + 1, gid, f"ADM_{defaults[country]}", union["city_level"],
                          level, tuple(row[0] for row in source_rows), parts,
                          union["operation"], union["attributes"]))
    if len(units) != RELEASE["units"]:
        raise ValueError("The selected city count differs from GUAB v1.0.")
    return units


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def wkb(blob):
    """Skip the fixed GeoPackage header and its optional coordinate envelope."""
    size = (0, 32, 48, 48, 64)[(blob[3] >> 1) & 7]
    return memoryview(blob)[8 + size:]


def gadm_rows(db, level, fids):
    layer = f"ADM_{level}"
    attributes = "COUNTRY, NULL, NULL" if level == 0 else f"NAME_{level}, TYPE_{level}, ENGTYPE_{level}"
    for start in range(0, len(fids), 256):
        keys = fids[start:start + 256]
        marks = ",".join("?" for _ in keys)
        yield from db.execute(f"""
            SELECT a.fid, a.GID_{level}, a.COUNTRY, {attributes}, a.geom,
                   b.minx, b.maxx, b.miny, b.maxy
            FROM {layer} a JOIN rtree_{layer}_geom b ON b.id = a.fid
            WHERE a.fid IN ({marks}) ORDER BY a.fid
        """, keys)


def feature_attributes(unit, row):
    _, _, country, name, unit_type, engtype = row[:6]
    attrs = dict(city_name=name, country=country, default_level=unit.default_level,
                 city_level=unit.city_level, gid=unit.gid, type=unit_type, engtype=engtype)
    if unit.level == 0:
        attrs.update(type="", engtype="")
    attrs.update(unit.overrides)
    attrs["unit_level"] = "city"
    attrs["source_gids"] = ";".join(unit.source_gids)
    attrs["boundary_source"] = f"GADM 4.1/ADM_{unit.level}"
    attrs["boundary_status"] = "approved_core" if unit.operation == "core_union" else "retained_v1"
    return attrs


def start_layer(target, db, units, spec):
    """Write the four unions; GDAL supplies the GeoPackage headers and spatial index."""
    records, shapes = [], []
    for unit in units:
        parts = {r[0]: r for r in gadm_rows(db, unit.level, unit.source_fids)}
        rows = [parts[k] for k in unit.source_fids]
        if tuple(r[1] for r in rows) != unit.source_gids:
            raise ValueError(f"GADM identifiers differ for {unit.gid}")
        geometry = shapely.union_all([shapely.from_wkb(bytes(wkb(r[6]))) for r in rows])
        if unit.operation == "v1_union" and geometry.geom_type == "Polygon":
            geometry = shapely.MultiPolygon([geometry])
        if hashlib.sha256(geometry.wkb).hexdigest() != spec["union_wkb_sha256"][unit.gid]:
            raise ValueError(f"Union coordinates differ from the reference for {unit.gid}")
        records.append(feature_attributes(unit, rows[0]))
        shapes.append(geometry)
    frame = gpd.GeoDataFrame(records, geometry=shapes, crs="EPSG:4326")
    pyogrio.write_dataframe(
        frame, target, layer="cities", driver="GPKG", geometry_type="Unknown",
        promote_to_multi=False, layer_options={"SPATIAL_INDEX": "YES", "GEOMETRY_NAME": "geom"},
        dataset_metadata={"title": "Global Urban Administrative Boundaries", "version": spec["version"]},
        layer_metadata={"unit": "One feature per study city", "source": "GADM 4.1",
                        "coordinate_policy": "Source coordinates retained, including boundary slivers."}
    )


def populate(target, adm, units, unions, version):
    levels = defaultdict(dict)
    for unit in units:
        if unit.operation == "direct":
            levels[unit.level][unit.source_fids[0]] = unit
    with closing(sqlite3.connect(target)) as db, db:
        # An ordinary SQLite connection cannot call GDAL's spatial trigger functions.
        # Copy each source R-tree envelope along with its unmodified geometry.
        triggers = db.execute("""
            SELECT name, sql FROM sqlite_master
            WHERE type = 'trigger' AND name LIKE 'rtree_cities_geom_%'
        """).fetchall()
        if len(triggers) < 4:
            raise ValueError("Missing spatial-index triggers")
        for name, _ in triggers:
            db.execute(f'DROP TRIGGER "{name}"')
        for unit in unions:
            old_fid, = db.execute("SELECT fid FROM cities WHERE gid=?", (unit.gid,)).fetchone()
            db.execute("UPDATE cities SET fid=? WHERE fid=?", (unit.fid, old_fid))
            db.execute("UPDATE rtree_cities_geom SET id=? WHERE id=?", (unit.fid, old_fid))

        columns = ",".join(("fid", "geom", *FIELDS))
        marks = ",".join("?" for _ in range(2 + len(FIELDS)))
        for level, selection in sorted(levels.items()):
            for row in gadm_rows(adm, level, sorted(selection)):
                unit = selection[row[0]]
                if unit.source_gids != (row[1],):
                    raise ValueError(f"GADM identifier differs for {unit.gid}")
                attrs = feature_attributes(unit, row)
                db.execute(f"INSERT INTO cities ({columns}) VALUES ({marks})",
                           (unit.fid, row[6], *(attrs[k] for k in FIELDS)))
                db.execute("INSERT INTO rtree_cities_geom VALUES (?,?,?,?,?)", (unit.fid, *row[7:]))
        for _, sql in triggers:
            db.execute(sql)
        bounds = db.execute("SELECT min(minx), min(miny), max(maxx), max(maxy) FROM rtree_cities_geom").fetchone()
        db.execute("""UPDATE gpkg_contents SET identifier=?, description=?,
                   min_x=?, min_y=?, max_x=?, max_y=? WHERE table_name='cities'""",
                   (f"GUAB v{version}", "City-level study boundaries, GADM 4.1, EPSG:4326.", *bounds))
        db.execute("UPDATE gpkg_ogr_contents SET feature_count=(SELECT count(*) FROM cities) WHERE table_name='cities'")
        db.execute("CREATE UNIQUE INDEX cities_gid ON cities (gid)")


def check_result(file, spec):
    with closing(readonly(file)) as db:
        layers = db.execute("SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns").fetchall()
        if layers != [("cities", "geom", 4326)]:
            raise ValueError(f"Unexpected layers or coordinate system: {layers}")
        count, unique_count = db.execute("SELECT count(*), count(DISTINCT gid) FROM cities").fetchone()
        index_count, = db.execute("SELECT count(*) FROM rtree_cities_geom").fetchone()
        indexed, = db.execute("SELECT count(*) FROM cities c JOIN rtree_cities_geom r ON r.id=c.fid").fetchone()
        if not (spec["units"] == count == unique_count == index_count == indexed):
            raise ValueError("Feature, ID and index counts disagree")
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)] or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("GeoPackage integrity check failed")
        fingerprint = hashlib.sha256()
        sql = f"SELECT geom, {','.join(FIELDS)} FROM cities ORDER BY gid COLLATE BINARY"
        for geometry, *values in db.execute(sql):
            fingerprint.update(hashlib.sha256(wkb(geometry)).digest())
            fingerprint.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
        if fingerprint.hexdigest() != spec["unit_digest"]:
            raise ValueError("City coordinates or attributes differ from GUAB v1.0.")


def build(gadm, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = RELEASE
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        draft = Path(folder) / "boundaries.gpkg"
        with closing(readonly(gadm)) as source:
            units = read_units(source)
            unions = [unit for unit in units if unit.operation != "direct"]
            start_layer(draft, source, unions, spec)
            populate(draft, source, units, unions, spec["version"])
        check_result(draft, spec)
        # Windows rename already refuses an existing destination. On POSIX use a hard link
        # for the same no-overwrite guarantee when two builds finish together.
        if os.name == "nt":
            draft.rename(output)
        else:
            os.link(draft, output)
    print(f"{len(units):,} city boundaries written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gadm", type=Path, required=True, help="GADM 4.1 levels GeoPackage")
    parser.add_argument("--output", type=Path, required=True, help="New output GeoPackage")
    args = parser.parse_args()
    build(args.gadm, args.output)
