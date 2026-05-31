"""Utilities for inserting inferred text into PAGE XML documents.

This module provides helpers for loading PAGE XML files, handling XML namespaces,
replacing direct ``TextEquiv`` elements, and writing OCR/HTR inference results
back to ``TextLine`` and ``TextRegion`` elements.
"""

# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import io
import xml.etree.ElementTree as et
from typing import Dict

from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================

class XMLProcessor:
    """Process PAGE XML documents and insert OCR/HTR inference results.

    The processor wraps an XML tree, keeps track of the document namespace, and
    provides helpers for creating and replacing ``TextEquiv`` elements. It can
    update matching ``TextLine`` elements with inferred text and add collected
    region-level text to the corresponding ``TextRegion`` elements.
    """
    def __init__(self, xml_file: str) -> None:
        """Load and parse a PAGE XML file.

        Args:
            xml_file: Path to the XML file to parse.

        Raises:
            FileNotFoundError: If the XML file does not exist.
            ValueError: If the XML file cannot be parsed.
        """
        try:
            self.tree = et.parse(xml_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"XML file '{xml_file}' not found.") from e
        except et.ParseError as e:
            raise ValueError(f"Error parsing XML file '{xml_file}': {e}") from e

        self.root = self.tree.getroot()

        # Safe namespace extraction
        if "}" in self.root.tag:
            self.namespace_uri = self.root.tag.split("}")[0][1:]
        else:
            self.namespace_uri = ""

        self.namespace = {"p": self.namespace_uri}
        self.xmlns = f"{{{self.namespace_uri}}}" if self.namespace_uri else ""

    @staticmethod
    def from_string(xml_content: str) -> "XMLProcessor":
        """Create an XMLProcessor instance from an XML string."""
        try:
            tree = et.ElementTree(et.fromstring(xml_content))
        except et.ParseError as e:
            raise ValueError(f"Error parsing XML content: {e}") from e

        instance = XMLProcessor.__new__(XMLProcessor)
        instance.tree = tree
        instance.root = tree.getroot()

        # Safe namespace extraction, same logic as __init__
        if "}" in instance.root.tag:
            instance.namespace_uri = instance.root.tag.split("}")[0][1:]
        else:
            instance.namespace_uri = ""

        instance.namespace = {"p": instance.namespace_uri}
        instance.xmlns = f"{{{instance.namespace_uri}}}" if instance.namespace_uri else ""

        return instance

    def create_text_equiv_element(self, text: str):
        """Create a TextEquiv element with one Unicode child."""
        text_equiv = et.Element(f"{self.xmlns}TextEquiv")
        unicode_el = et.SubElement(text_equiv, f"{self.xmlns}Unicode")
        unicode_el.text = text
        return text_equiv

    def replace_direct_text_equiv(self, element, text: str) -> None:
        """Replace only direct TextEquiv children of the given element.

        This is used for both TextLine and TextRegion. It intentionally does not
        touch nested TextEquiv elements.
        """
        for old_text_equiv in element.findall(f"{self.xmlns}TextEquiv"):
            element.remove(old_text_equiv)

        element.append(self.create_text_equiv_element(text))

    def insert_inferred_lines(
        self,
        root,
        inferred_lines: Dict[tuple[str, str], str],
    ) -> int:
        """Insert inferred text into PAGE XML.

        Expected input:

            inferred_lines = {
                (region_id, line_id): text
            }

        Behavior:
        - Updates matching TextLine elements with a direct TextEquiv.
        - Writes one direct TextEquiv at the end of each updated TextRegion.
        - The TextRegion TextEquiv contains the collected text of all updated
          TextLine elements in that region, joined with newlines.
        - Existing direct TextEquiv elements on updated TextLine/TextRegion
          elements are replaced.
        - Nested TextEquiv elements are not touched.
        - Duplicate (region_id, line_id) pairs in the XML are skipped after the
          first occurrence.
        - Empty predictions are skipped so existing XML text is not erased.

        Returns:
            Number of TextLine elements updated.
        """
        updated_count = 0
        seen_keys: set[tuple[str, str]] = set()

        for region in root.findall(f".//{self.xmlns}TextRegion"):
            region_id = region.get("id")

            if not region_id:
                continue

            region_texts: list[str] = []

            for text_line in region.findall(f"{self.xmlns}TextLine"):
                line_id = text_line.get("id")

                if not line_id:
                    continue

                key = (str(region_id), str(line_id))

                if key in seen_keys:
                    logger.warning(
                        f"Duplicate TextLine key in XML: "
                        f"region_id={region_id}, line_id={line_id}. "
                        "Skipping duplicate."
                    )
                    continue

                seen_keys.add(key)

                if key not in inferred_lines:
                    continue

                text = str(inferred_lines[key]).strip()

                if not text:
                    continue

                self.replace_direct_text_equiv(text_line, text)
                updated_count += 1
                region_texts.append(text)

            region_text = "\n".join(region_texts)
            self.replace_direct_text_equiv(region, region_text)

        return updated_count

    def tree_to_string(self) -> str:
        """Serialize the XML tree to a string."""
        if self.namespace_uri:
            et.register_namespace("", self.namespace_uri)

        buf = io.StringIO()
        self.tree.write(buf, encoding="unicode", xml_declaration=True)
        return buf.getvalue()