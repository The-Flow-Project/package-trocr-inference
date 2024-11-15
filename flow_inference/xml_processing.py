import xml.etree.ElementTree as et
from typing import Dict


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

    def insert_text_equiv_tags(self, root, inferred_lines: Dict[str, str]):
        """
        Inserts a <TextEquiv><Unicode>...</Unicode></TextEquiv> tag under each <TextLine>
        element that matches an entry in the inferred_lines dictionary.

        Args:
            root (ElementTree.Element): Root element of the XML tree.
            inferred_lines (dict): Dictionary with labels and text.
        """
        for text_line in root.findall(f".//{self.xmlns}TextLine"):  # Iterate over <TextLine> elements
            line_id = text_line.get("id")  # Extract the 'id' attribute of the <TextLine>
            if line_id in inferred_lines:
                # Create the <TextEquiv> element
                text_equiv = self.create_text_equiv_element(inferred_lines[line_id])
                # Append the <TextEquiv> element to the <TextLine>
                text_line.append(text_equiv)

    @staticmethod
    def save_xml(tree, output_path):
        """
        Saves the modified XML tree to a file.

        Args:
            tree (ElementTree.ElementTree): The XML tree.
            output_path (str): Path to save the XML file.
        """
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
