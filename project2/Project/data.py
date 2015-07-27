#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
from audit_address import update_type, update_beginning, is_street_name
from audit_phone import update_phone, is_phone
from audit_postcode import update_postcode, is_postcode

""" Wrangle the data and transform the shape of the data according to data model in lesson 6. The output should be a list of dictionaries that look like this:

{
"id": "2406124091",
"type: "node",
"visible":"true",
"created": {
          "version":"2",
          "changeset":"17206049",
          "timestamp":"2013-08-03T16:43:42Z",
          "user":"linuxUser16",
          "uid":"1219059"
        },
"pos": [41.9757030, -87.6921867],
"address": {
          "housenumber": "5157",
          "postcode": "60625",
          "street": "North Lincoln Ave"
        },
"amenity": "restaurant",
"cuisine": "mexican",
"name": "La Cabana De Don Luis",
"phone": "1 (773)-271-5176"
}
"""

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]




def shape_element(element):
    node, created , loc, address, ref_list = {}, {}, [], {}, []
    # Process only 2 types of top level tags: "node" and "way".
    if element.tag == "node" or element.tag == "way" :
     
        # Attributes id, tag type, visible of "node" and "way" are turned into regular key/value pairs.
        node["id"] = element.attrib["id"]
        node["type"] = element.tag
        if "visible" in element.attrib:
            node["visible"] = element.attrib["visible"]
        
        # Attributes in the CREATED array should be added under a key "created"
            
        for i in CREATED:
            created[i] = element.attrib[i]
     
        node["created"] = created
        
            
        # Attributes for latitude and longitude are added as floats to a "pos" array.
       
        if element.tag == 'node':
            loc.append(float(element.attrib["lat"]))
            loc.append(float(element.attrib["lon"]))
            node["pos"] = loc
        else:
            for tag in element.iter("nd"):
                ref_list.append(tag.attrib['ref'])
            node["node_refs"] = ref_list
            
            
        for tag in element.iter("tag"):
            # If second level tag "k" value contains problematic characters, it should be ignored.
            if problemchars.search(tag.attrib['k']):
                continue
            
            # If second level tag "k" value starts with "addr:", it should be added to a dictionary "address"
            
            if tag.attrib['k'][0:5] == "addr:":
                if "street:" in tag.attrib['k'][5:]:
                    continue
                if is_street_name(tag):
                    tag.attrib['v'] = update_type(tag.attrib['v'])
                    tag.attrib['v'] = update_beginning(tag.attrib['v'])
                elif is_postcode(tag):
                    tag.attrib['v'] = update_postcode(tag.attrib['v'])
                address[tag.attrib['k'][5:]] = tag.attrib['v']
            else:
                if is_phone(tag):
                    tag.attrib['v'] = update_phone(tag.attrib['v'])
                node[tag.attrib['k']] = tag.attrib['v']
            
        if address != {}:
            node['address'] = address

        return node
    else:
        return None


def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

def test():
    # NOTE: if you are running this code on your computer, with a larger dataset, 
    # call the process_map procedure with pretty=False. The pretty=True option adds 
    # additional spaces to the output, making it significantly larger.
    data = process_map('mobile_alabama.osm', True)
    #pprint.pprint(data)
    
  
if __name__ == "__main__":
    test()


"""
created["version"] = element.attrib["version"]
created["changeset"] = element.attrib["changeset"]
created["timestamp"] = element.attrib["timestamp"]
created["user"] = element.attrib["user"]
created["uid"] = element.attrib["uid"]
"""
