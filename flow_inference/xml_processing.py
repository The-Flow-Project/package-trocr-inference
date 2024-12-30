import xml.etree.ElementTree as et
from typing import Dict
from xml.dom import minidom


class XMLProcessor:
    def __init__(self, xml_file):
        self.tree = et.parse(xml_file)
        self.root = self.tree.getroot()
        self.namespace_uri = self.root.tag.split('}')[0][1:]
        self.namespace = {'prefix': self.namespace_uri}
        self.xmlns = '{' + self.namespace_uri + '}'

    @staticmethod
    def parse_xml(xml_path):
        """
        Parses an XML file and returns the root element.

        Args:
            xml_path (str): Path to the XML file.

        Returns:
            ElementTree.Element: Root element of the XML tree.
        """
        tree = et.parse(xml_path)
        return tree

    @staticmethod
    def find_line_id(text_line):
        """
        Extracts the line ID from a <TextLine> element.

        Args:
            text_line (ElementTree.Element): The <TextLine> element.

        Returns:
            str: The extracted line ID, or None if not found.
        """
        return text_line.get('id')

    def create_text_equiv_element(self, text):
        """
        Creates a <TextEquiv> element with a <Unicode> child containing the provided text.

        Args:
            text (str): The text to insert into the <Unicode> element.

        Returns:
            ElementTree.Element: The created <TextEquiv> element.
        """
        # Create <TextEquiv> and <Unicode> with namespace
        ns_tag_text_equiv = f"{self.xmlns}TextEquiv"
        ns_tag_unicode = f"{self.xmlns}Unicode"

        text_equiv = et.Element(ns_tag_text_equiv)
        unicode_element = et.SubElement(text_equiv, ns_tag_unicode)
        unicode_element.text = text
        return text_equiv

    def insert_inferred_lines(self, root, inferred_lines: Dict[str, str]):
        """
        Inserts a <TextEquiv><Unicode>...</Unicode></TextEquiv> tag under each <TextLine>
        element that matches an entry in the inferred_lines dictionary.

        Args:
            root (ElementTree.Element): Root element of the XML tree.
            inferred_lines (dict): Dictionary with labels and text.
        """
        modified_inferred_lines = {key.split('.')[1] if '.' in key else key: value for key, value in
                                   inferred_lines.items()}
        for text_line in root.findall(f".//{self.xmlns}TextLine"):  # Iterate over <TextLine> elements
            line_id = text_line.get("id")  # Extract the 'id' attribute of the <TextLine>
            if line_id in modified_inferred_lines.keys():
                # Create the <TextEquiv> element
                text_equiv = self.create_text_equiv_element(modified_inferred_lines[line_id])
                # Append the <TextEquiv> element to the <TextLine>
                text_line.append(text_equiv)

    @staticmethod
    def save_xml(tree, output_path):
        """
        Saves the modified XML tree to a file with pretty-printing.

        Args:
            tree (ElementTree.ElementTree): The XML tree.
            output_path (str): Path to save the XML file.
        """
        # Convert the ElementTree to a string
        xml_str = et.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")

        # Pretty print the XML using minidom
        xml_str_pretty = minidom.parseString(xml_str).toprettyxml(indent="    ")  # 4 spaces indentation

        # Remove extra blank lines from the pretty-printed string
        xml_str_pretty = "\n".join([line for line in xml_str_pretty.splitlines() if line.strip()])

        # Write the cleaned up, pretty-printed XML string to the file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str_pretty)
