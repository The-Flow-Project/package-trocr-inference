# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import xml.etree.ElementTree as et
from typing import Dict
import io


# ===============================================================================
# CLASS
# ===============================================================================
class XMLProcessor:
    def __init__(self, xml_file: str) -> None:
        """
        Initializes the XMLProcessor with the specified XML file.
        """
        try:
            self.tree = et.parse(xml_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"XML file '{xml_file}' not found.") from e
        except et.ParseError as e:
            raise ValueError(f"Error parsing XML file '{xml_file}': {e}") from e

        self.root = self.tree.getroot()

        # safe namespace extraction
        if "}" in self.root.tag:
            self.namespace_uri = self.root.tag.split('}')[0][1:]
        else:
            self.namespace_uri = ""

        self.namespace = {'p': self.namespace_uri}
        self.xmlns = f"{{{self.namespace_uri}}}" if self.namespace_uri else ""

    @staticmethod
    def from_string(xml_content: str) -> "XMLProcessor":
        tree = et.ElementTree(et.fromstring(xml_content))
        instance = XMLProcessor.__new__(XMLProcessor)
        instance.tree = tree
        instance.root = tree.getroot()

        # safe namespace extraction (same logic as __init__)
        if "}" in instance.root.tag:
            instance.namespace_uri = instance.root.tag.split('}')[0][1:]
        else:
            instance.namespace_uri = ""

        instance.namespace = {'p': instance.namespace_uri}
        instance.xmlns = f"{{{instance.namespace_uri}}}" if instance.namespace_uri else ""
        return instance

    def create_text_equiv_element(self, text: str):
        te = et.Element(f"{self.xmlns}TextEquiv")
        u = et.SubElement(te, f"{self.xmlns}Unicode")
        u.text = text
        return te

    def insert_inferred_lines(self, root, inferred_lines: Dict[str, str]):
        """
        inferred_lines = { line_id: text }
        """
        for tl in root.findall(f".//{self.xmlns}TextLine"):
            tl_id = tl.get("id")
            if tl_id in inferred_lines:
                # delete existing TextEquiv
                for old_te in tl.findall(f"{self.xmlns}TextEquiv"):
                    tl.remove(old_te)

                # insert new one
                tl.append(self.create_text_equiv_element(inferred_lines[tl_id]))

    def extract_all_text_lines(self):
        lines = []
        for tl in self.root.findall(f".//{self.xmlns}TextLine"):
            te = tl.find(f"{self.xmlns}TextEquiv")
            if te is not None:
                u = te.find(f"{self.xmlns}Unicode")
                if u is not None and u.text:
                    lines.append(u.text.strip())
        return lines

    def tree_to_string(self) -> str:
        et.register_namespace("", self.namespace_uri)
        buf = io.StringIO()
        self.tree.write(buf, encoding="unicode", xml_declaration=True)
        return buf.getvalue()

