# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import xml.etree.ElementTree as et
from typing import Dict, Optional
from xml.dom import minidom
from xml.etree.ElementTree import ElementTree, Element

import pandas as pd


# ===============================================================================
# CLASS
# ===============================================================================
class XMLProcessor:
    def __init__(self, xml_file: str) -> None:
        """
        Initializes the XMLProcessor with the specified XML file.

        :param: xml_file (str): Path to the XML file to process.
        :returns: None.
        """
        try:
            self.tree = et.parse(xml_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"XML file '{xml_file}' not found.") from e
        except et.ParseError as e:
            raise ValueError(f"Error parsing XML file '{xml_file}': {e}") from e

        self.root = self.tree.getroot()
        self.namespace_uri = self.root.tag.split('}')[0][1:]
        self.namespace = {'prefix': self.namespace_uri}
        self.xmlns = f'{{{self.namespace_uri}}}'

    @staticmethod
    def from_string(xml_content: str) -> "XMLProcessor":
        """
        Create an XMLProcessor instance from a raw XML string instead of a file path.
        """
        import xml.etree.ElementTree as et

        try:
            tree = et.ElementTree(et.fromstring(xml_content))
            instance = XMLProcessor.__new__(XMLProcessor)
            instance.tree = tree
            instance.root = tree.getroot()
            instance.namespace_uri = instance.root.tag.split('}')[0][1:]
            instance.namespace = {'prefix': instance.namespace_uri}
            instance.xmlns = f'{{{instance.namespace_uri}}}'
            return instance
        except et.ParseError as e:
            raise ValueError(f"Failed to parse XML string: {e}") from e

    @staticmethod
    def parse_xml(xml_path: str) -> ElementTree:
        """
        Parses an XML file and returns the root element.

        :param: Path to the XML file.
        :returns: Root element of the XML tree.
        """
        try:
            return et.parse(xml_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File '{xml_path}' not found.") from e
        except et.ParseError as e:
            raise ValueError(f"Failed to parse XML file '{xml_path}': {e}") from e

    @staticmethod
    def find_line_id(text_line: Element) -> Optional[str]:
        """
        Extracts the line ID from a <TextLine> element.

        :param: text_line (ElementTree.Element): The <TextLine> element.
        :returns: The extracted line ID, or None if not found.
        """
        return text_line.get('id')

    def create_text_equiv_element(self, text: str) -> Element:
        """
        Creates a <TextEquiv> element with a <Unicode> child containing the provided text.

        :param: The text to insert into the <Unicode> element.

        :returns: ElementTree.Element: The created <TextEquiv> element.
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
                text_equiv = self.create_text_equiv_element(modified_inferred_lines[line_id])
                text_line.append(text_equiv)

    @staticmethod
    def save_xml(tree: ElementTree, output_path: str) -> None:
        """
        Saves the modified XML tree to a file with pretty-printing.

        :param: The XML tree.
        :param: Path to save the XML file.
        """
        try:
            xml_str = et.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")
            xml_str_pretty = minidom.parseString(xml_str).toprettyxml(indent="    ")
            xml_str_pretty = "\n".join(line for line in xml_str_pretty.splitlines() if line.strip())

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(xml_str_pretty)
        except IOError as e:
            raise IOError(f"Error saving XML to '{output_path}': {e}") from e

    def extract_text_from_textline(self, text_line: Element) -> Optional[str]:
        """
        Extracts text from a given <TextLine> element.

        Args:
            text_line (Element): The <TextLine> XML element.

        Returns:
            str or None: Extracted text or None if <Unicode> is missing.
        """
        text_equiv = text_line.find(f"{self.xmlns}TextEquiv")
        if text_equiv is not None:
            unicode_elem = text_equiv.find(f"{self.xmlns}Unicode")
            return unicode_elem.text.strip() if unicode_elem is not None and unicode_elem.text else None
        return None

    def extract_all_text_lines(self) -> list:
        """
        Extracts text content from all <TextLine> elements in the XML.

        Returns:
            list: List of extracted text strings (empty if no text exists).
        """
        text_lines = self.root.findall(f".//{self.xmlns}TextLine")
        return [self.extract_text_from_textline(line) for line in text_lines if self.extract_text_from_textline(line)]

    def update_raw_xml_in_records(
            self,
            inferred_lines: Dict[str, str],
            original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Update the raw XML column in the DataFrame with its corresponding inference results.
        Returns the updated DataFrame.
        """

        updated_df = original_df.copy()
        updated_count = 0

        for idx, row in updated_df.iterrows():
            filename = row.get("filename")
            raw_xml = row.get("xml")

            if not raw_xml:
                logger.debug(f"No XML found for {filename}. Skipping.")
                continue

            inferred_text = inferred_lines.get(filename)
            if not inferred_text:
                continue

            try:
                xml_processor = XMLProcessor.from_string(raw_xml)
                xml_processor.insert_inferred_lines(
                    root=xml_processor.root,
                    inferred_lines={filename: inferred_text}
                )

                # Convert updated XML tree back to string
                import io
                xml_str = io.StringIO()
                xml_processor.tree.write(xml_str, encoding="unicode")
                updated_df.at[idx, "xml"] = xml_str.getvalue()
                updated_count += 1

            except Exception as e:
                logger.error(f"Failed to update XML for {filename}: {e}")

        logger.info(f"Updated XML for {updated_count} records successfully.")
        return updated_df
