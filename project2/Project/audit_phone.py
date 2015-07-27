import xml.etree.cElementTree as ET
import re
import codecs
import pprint
import json

OSMFILE = "mobile_alabama.osm"

def is_phone(elem):
    return (elem.attrib['k'] == "phone")

def audit_phone(osmfile):
    """Find all phone numbers and return them in a list """
    phone_list = []
    osm_file = open(osmfile, "r")
    for event, element in ET.iterparse(osm_file, events=("start",)):
        if element.tag == "node" or element.tag == "way":
            for tag in element.iter("tag"):
                if is_phone(tag):
                    phone_list.append(tag.attrib['v'])
                
    return phone_list

def update_phone(phone):
    """ Formats phone number according to North American Numbering Plan for national and international phone numbers. """
    
    count = 0 # Count the number of digits in the string phone
    for char in phone:
        if char.isdigit():
            count = count +1
        else:
            continue # Ignore all non-digits.
        
        # If count = 10 the phone number national and should
        # hat format 2xx-2xx-xxx. Format according the NANP for national phonenumbers.
        if count == 10 and re.search(r"[^a-zA-Z0-9_]", phone):
            updated_phone = re.sub(r"[^a-zA-Z0-9_]","", phone)
            updated_phone = updated_phone[0:3] + "-"+ updated_phone[3:6] + "-" + updated_phone[6:]
            
        # if count = 11 the phone number international and should
        # has format +1-2xx-2xx-xxx. Format according the NANP for international phone numbers.
        elif count == 11 and re.search(r"[^a-zA-Z0-9_]", phone):
            re.search(r"[^a-zA-Z0-9_]", phone)
            updated_phone = re.sub(r"[^a-zA-Z0-9_]","", phone)
            updated_phone = "+"+updated_phone[0]+"-"+ updated_phone[1:4] + "-"+ updated_phone[4:7] + "-" + updated_phone[6:]
            
        else:
            updated_phone = phone

    return updated_phone



def test():
    audit_list = audit_phone(OSMFILE)
    print audit_list

    for i in audit_list:
        print update_phone(i)

if __name__ == '__main__':
    test()

            
                    


