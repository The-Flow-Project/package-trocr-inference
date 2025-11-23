import unittest
import xml.etree.ElementTree as ET
import os

from flow_inference.xml_processing import XMLProcessor


# A small inline PAGE XML for testing (namespace + 3 lines)
SAMPLE_XML = """
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
    <Page imageFilename="sample.png">
        <TextRegion id="r1">
            <TextLine id="L1">
                <Coords points="0,0 10,0 10,10 0,10"/>
            </TextLine>
            <TextLine id="L2">
                <Coords points="0,20 10,20 10,30 0,30"/>
            </TextLine>
            <TextLine id="L3">
                <Coords points="0,40 10,40 10,50 0,50"/>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>
"""


class TestXMLProcessor(unittest.TestCase):

    def test_parse_xml(self):
        """Test parsing an XML string via from_string."""
        xp = XMLProcessor.from_string(SAMPLE_XML)
        self.assertIsNotNone(xp.root)

    def test_create_text_equiv_element(self):
        xp = XMLProcessor.from_string(SAMPLE_XML)

        el = xp.create_text_equiv_element("Hello")
        # Namespaced tag
        self.assertTrue(el.tag.endswith("TextEquiv"))

        uni = el.find(f"{xp.xmlns}Unicode")
        self.assertIsNotNone(uni)
        self.assertEqual(uni.text, "Hello")

    def test_insert_inferred_lines(self):
        xp = XMLProcessor.from_string(SAMPLE_XML)

        inferred = {
            "L1": "Line 1 text",
            "L3": "Line 3 text"
        }

        xp.insert_inferred_lines(xp.root, inferred)

        ns = {"ns": xp.namespace_uri}
        updated = ET.ElementTree(ET.fromstring(xp.tree_to_string()))
        root = updated.getroot()

        # Check L1
        L1 = root.find(".//ns:TextLine[@id='L1']", namespaces=ns)
        te1 = L1.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(te1)
        self.assertEqual(
            te1.find(".//ns:Unicode", namespaces=ns).text,
            "Line 1 text"
        )

        # L2 should NOT have TextEquiv
        L2 = root.find(".//ns:TextLine[@id='L2']", namespaces=ns)
        te2 = L2.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNone(te2)

        # L3 should have TextEquiv
        L3 = root.find(".//ns:TextLine[@id='L3']", namespaces=ns)
        te3 = L3.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(te3)
        self.assertEqual(
            te3.find(".//ns:Unicode", namespaces=ns).text,
            "Line 3 text"
        )

    def test_tree_to_string(self):
        xp = XMLProcessor.from_string(SAMPLE_XML)
        xml_string = xp.tree_to_string()

        self.assertIsInstance(xml_string, str)
        self.assertIn("TextLine", xml_string)


if __name__ == "__main__":
    unittest.main()
