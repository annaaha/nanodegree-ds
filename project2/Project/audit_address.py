"""
Your task in this exercise has two steps:

- audit the OSMFILE and change the variable 'mapping' to reflect the changes needed to fix 
    the unexpected street types to the appropriate ones in the expected list.
    You have to add mappings only for the actual problems you find in this OSMFILE,
    not a generalized solution, since that may and will depend on the particular area you are auditing.
- write the update_name function, to actually fix the street name.
    The function takes a string with street name as an argument and should return the fixed name
    We have provided a simple test so that you see what exactly is expected
"""
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "mobile_alabama.osm"
#Regular expressions
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
street_name_beginning_re = re.compile(r'^\b\S+\.?', re.IGNORECASE)

# Expected street types
expected_street_type = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", "Trail", "Parkway", "Commons", "Way", "North","South", "West", "East", "Laurel", "Highway", "Crossing", "Circle", "Causeway"]

# Mapping street types
mapping_street_type = {
    "Ave" : "Avenue",
    "Blvd" : "Boulevard",
    "N" : "North",
    "S" : "South",
    "W" : "West",
    "E" : "East",
    "Dr" : "Drive"
    }

# Mappring street beginning abbreviations
mapping_street_name_beginning = {
    "Hwy." : "Highway",
    "S" : "South",
    "Mc ": "Mc",
    "Dr" : "Dr.",
    "N." : "North",
    "Capt." : "Captain"
    }
    


def audit_street(street_types,street_names_beginning, street_name):
    """ Add a street name to respective set in audit dicts. """
    # m is the last word of a street name.
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected_street_type:
            street_types[street_type].add(street_name)
            
    # l is the first word of a street name.        
    l = street_name_beginning_re.search(street_name)
    if l:
        street_name_beginning = l.group()
        if street_name_beginning in mapping_street_name_beginning.keys():
            street_names_beginning[street_name_beginning].add(street_name)
            

def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit_street_name(osmfile):
    """Audit street names. """
    osm_file = open(osmfile, "r")
    
    """ Audit dict for street types. Keys are abbreviations
    at the ending of the street name which has to be replaced.
    Values are the corresponding set of street names. """
    street_types = defaultdict(set)

    """ Audit dict for street name beginning. Keys are abbreviations
    at the beginning of the street name which has to be replaced. 
    Values are the corresponding set of street names. """
    street_names_beginning = defaultdict(set)

    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street(street_types, street_names_beginning, tag.attrib['v'])

    return [street_types, street_names_beginning]


def update_type(name):
    """Updates the street name if the last word of it is an abbreviation."""
    if street_type_re.search(name).group() in mapping_street_type.keys():
        name = street_type_re.sub(mapping_street_type[street_type_re.search(name).group()], name)
   
    return name


def update_beginning(name):
    """Updates the street name if the first word of it is an abbreviation."""
    if street_name_beginning_re.search(name).group() in mapping_street_name_beginning.keys():
        name = street_name_beginning_re.sub(mapping_street_name_beginning[street_name_beginning_re.search(name).group()], name)

    return name


def test():
    st_types = audit_street_name(OSMFILE)
    print st_types    
       
    beginnings = []
    for i in st_types[1].values():
        for names in i:
            beginnings.append(names)
    #print (dict(st_types))

    #assert len(st_types) == 3

    for st_type, ways in st_types[0].iteritems():
        for name in ways:
            better_name = update_type(name)
            print name, "=>", better_name
            if name in beginnings:
                beginnings[beginnings.index(name)] = better_name
                
    for name in beginnings:
        better_name = update_beginning(name)
        print name, "=>", better_name

if __name__ == '__main__':
    test()

