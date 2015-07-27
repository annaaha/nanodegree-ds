import xml.etree.cElementTree as ET
import re
import codecs
import pprint
import json

OSMFILE = "mobile_alabama.osm"

postcodes =str([36601, 36602, 36603, 36604, 36605, 36606, 36607, 36608,36609, 36610, 36611, 36612, 36615, 36616, 36617, 36618, 36619, 36625, 36628, 36630, 36633, 36640,36641, 36644, 36652, 36660, 36663, 36670, 36671, 36675, 36685, 36688, 36689,36691, 36693, 36695])

codes_re = re.compile(r"\D")


def is_postcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit_postcode(osmfile):
    """Find all phone numbers and return them in a list """
    not_mobile_postcode = []
    osm_file = open(osmfile, "r")
    for event, element in ET.iterparse(osm_file, events=("start",)):
        if element.tag == "node" or element.tag == "way":
                for tag in element.iter("tag"):
                    if is_postcode(tag):
                        if tag.attrib['v'] not in postcodes:
                            not_mobile_postcode.append(tag.attrib['v'])

    return not_mobile_postcode
    
                    
def update_postcode(postcode):
    """ Update the postcodes by removing all non digits. """
    updated_postcode = codes_re.sub("", str(postcode))

    return updated_postcode


def test():
    audit_list = audit_postcode(OSMFILE)
    print audit_list

    for i in audit_list:
        print update_postcode(i)

if __name__ == '__main__':
    test()


"""
	
Not postcodes set(['36613', '36527', '36575', '36532', '36526'])

postcode: "36613", EIGHT MILE, AL,  city" v="Eight Mile", v="High Pointe Golf Course", within 15 miles of MOBILE, AL
postcode: "36527, Spanish Fort, AL, way
postcode: '36575', Semmes, AL, relation
postcode: "36532": Fairhope, AL, in node
postcode: '36526': Daphne, AL, in node


"mobile_alabama": {
                    "bbox": {
                        "top": "30.883",
                        "left": "-88.363",
                        "bottom": "30.496",
                        "right": "-87.832"
                    }


"""
