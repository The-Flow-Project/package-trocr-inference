import unittest
import xml.etree.ElementTree as ET
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
        self.assertTrue(el.tag.endswith("TextEquiv"))

        uni = el.find(f"{xp.xmlns}Unicode")
        self.assertIsNotNone(uni)
        self.assertEqual(uni.text, "Hello")

    def test_insert_inferred_lines(self):
        xp = XMLProcessor.from_string(SAMPLE_XML)

        inferred = {
            ("r1", "L1"): "Line 1 text",
            ("r1", "L3"): "Line 3 text",
        }

        updated_count = xp.insert_inferred_lines(xp.root, inferred)
        self.assertEqual(updated_count, 2)

        ns = {"ns": xp.namespace_uri}
        updated = ET.ElementTree(ET.fromstring(xp.tree_to_string()))
        root = updated.getroot()

        # Check L1
        L1 = root.find(".//ns:TextLine[@id='L1']", namespaces=ns)
        te1 = L1.find("ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(te1)
        self.assertEqual(
            te1.find("ns:Unicode", namespaces=ns).text,
            "Line 1 text",
        )

        # L2 should NOT have TextEquiv
        L2 = root.find(".//ns:TextLine[@id='L2']", namespaces=ns)
        te2 = L2.find("ns:TextEquiv", namespaces=ns)
        self.assertIsNone(te2)

        # L3 should have TextEquiv
        L3 = root.find(".//ns:TextLine[@id='L3']", namespaces=ns)
        te3 = L3.find("ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(te3)
        self.assertEqual(
            te3.find("ns:Unicode", namespaces=ns).text,
            "Line 3 text",
        )

        # Region should have one direct TextEquiv with collected line texts
        region = root.find(".//ns:TextRegion[@id='r1']", namespaces=ns)
        region_text_equivs = region.findall("ns:TextEquiv", namespaces=ns)

        self.assertEqual(len(region_text_equivs), 1)
        self.assertEqual(
            region_text_equivs[0].find("ns:Unicode", namespaces=ns).text,
            "Line 1 text\nLine 3 text",
        )

    def test_insert_inferred_lines_replaces_existing_region_text_equiv(self):
        xml_with_existing_region_text_equiv = """
        <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
            <Page imageFilename="sample.png">
                <TextRegion id="r1">
                    <TextLine id="L1">
                        <Coords points="0,0 10,0 10,10 0,10"/>
                    </TextLine>
                    <TextEquiv>
                        <Unicode></Unicode>
                    </TextEquiv>
                </TextRegion>
            </Page>
        </PcGts>
        """

        xp = XMLProcessor.from_string(xml_with_existing_region_text_equiv)

        updated_count = xp.insert_inferred_lines(
            xp.root,
            {("r1", "L1"): "Replacement region text"},
        )

        self.assertEqual(updated_count, 1)

        ns = {"ns": xp.namespace_uri}
        root = ET.fromstring(xp.tree_to_string())

        region = root.find(".//ns:TextRegion[@id='r1']", namespaces=ns)
        region_text_equivs = region.findall("ns:TextEquiv", namespaces=ns)

        self.assertEqual(len(region_text_equivs), 1)
        self.assertEqual(
            region_text_equivs[0].find("ns:Unicode", namespaces=ns).text,
            "Replacement region text",
        )

    def test_insert_inferred_lines_adds_region_text_equiv_when_missing(self):
        xp = XMLProcessor.from_string(SAMPLE_XML)

        updated_count = xp.insert_inferred_lines(
            xp.root,
            {("r1", "L1"): "New region text"},
        )

        self.assertEqual(updated_count, 1)

        ns = {"ns": xp.namespace_uri}
        root = ET.fromstring(xp.tree_to_string())

        region = root.find(".//ns:TextRegion[@id='r1']", namespaces=ns)
        region_text_equivs = region.findall("ns:TextEquiv", namespaces=ns)

        self.assertEqual(len(region_text_equivs), 1)
        self.assertEqual(
            region_text_equivs[0].find("ns:Unicode", namespaces=ns).text,
            "New region text",
        )

    def test_tree_to_string(self):
        xp = XMLProcessor.from_string(SAMPLE_XML)
        xml_string = xp.tree_to_string()

        self.assertIsInstance(xml_string, str)
        self.assertIn("TextLine", xml_string)

    def test_insert_inferred_lines_skips_duplicate_textline_ids_in_same_region(self):
        xml_with_duplicate_id = """
        <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
            <Page imageFilename="sample.png">
                <TextRegion id="r1">
                    <TextLine id="L1">
                        <Coords points="0,0 10,0 10,10 0,10"/>
                    </TextLine>
                    <TextLine id="L1">
                        <Coords points="0,20 10,20 10,30 0,30"/>
                    </TextLine>
                </TextRegion>
            </Page>
        </PcGts>
        """

        xp = XMLProcessor.from_string(xml_with_duplicate_id)

        updated_count = xp.insert_inferred_lines(
            xp.root,
            {("r1", "L1"): "Line 1 text"},
        )
        self.assertEqual(updated_count, 1)

        ns = {"ns": xp.namespace_uri}
        root = ET.fromstring(xp.tree_to_string())
        lines = root.findall(".//ns:TextLine[@id='L1']", namespaces=ns)

        self.assertEqual(len(lines), 2)

        first_te = lines[0].find("ns:TextEquiv", namespaces=ns)
        second_te = lines[1].find("ns:TextEquiv", namespaces=ns)

        self.assertIsNotNone(first_te)
        self.assertIsNone(second_te)

        region = root.find(".//ns:TextRegion[@id='r1']", namespaces=ns)
        region_te = region.find("ns:TextEquiv", namespaces=ns)

        self.assertIsNotNone(region_te)
        self.assertEqual(
            region_te.find("ns:Unicode", namespaces=ns).text,
            "Line 1 text",
        )

    def test_insert_inferred_lines_returns_zero_when_no_ids_match(self):
        xp = XMLProcessor.from_string(SAMPLE_XML)

        updated_count = xp.insert_inferred_lines(
            xp.root,
            {("r1", "DOES_NOT_EXIST"): "Text"},
        )
        self.assertEqual(updated_count, 0)

        ns = {"ns": xp.namespace_uri}
        root = ET.fromstring(xp.tree_to_string())

        # No TextLine should receive TextEquiv.
        line_text_equivs = root.findall(".//ns:TextLine/ns:TextEquiv", namespaces=ns)
        self.assertEqual(len(line_text_equivs), 0)

        # But each TextRegion should now have one direct empty TextEquiv.
        region = root.find(".//ns:TextRegion[@id='r1']", namespaces=ns)
        self.assertIsNotNone(region)

        region_te = region.find("ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(region_te)

        unicode_el = region_te.find("ns:Unicode", namespaces=ns)
        self.assertIsNotNone(unicode_el)
        self.assertIn(unicode_el.text, (None, ""))

    def test_insert_inferred_lines_uses_region_id_to_disambiguate_same_line_id(self):
        xml_with_same_line_id_in_different_regions = """
        <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
            <Page imageFilename="sample.png">
                <TextRegion id="r1">
                    <TextLine id="L1">
                        <Coords points="0,0 10,0 10,10 0,10"/>
                    </TextLine>
                </TextRegion>
                <TextRegion id="r2">
                    <TextLine id="L1">
                        <Coords points="0,20 10,20 10,30 0,30"/>
                    </TextLine>
                </TextRegion>
            </Page>
        </PcGts>
        """

        xp = XMLProcessor.from_string(xml_with_same_line_id_in_different_regions)

        updated_count = xp.insert_inferred_lines(
            xp.root,
            {("r2", "L1"): "Only region 2 should be updated"},
        )

        self.assertEqual(updated_count, 1)

        ns = {"ns": xp.namespace_uri}
        root = ET.fromstring(xp.tree_to_string())

        r1 = root.find(".//ns:TextRegion[@id='r1']", namespaces=ns)
        r2 = root.find(".//ns:TextRegion[@id='r2']", namespaces=ns)

        r1_l1 = r1.find("ns:TextLine[@id='L1']", namespaces=ns)
        r2_l1 = r2.find("ns:TextLine[@id='L1']", namespaces=ns)

        self.assertIsNotNone(r1_l1)
        self.assertIsNotNone(r2_l1)

        # r1 line is not updated.
        self.assertIsNone(r1_l1.find("ns:TextEquiv", namespaces=ns))

        # But r1 region still receives an empty region-level TextEquiv.
        r1_region_te = r1.find("ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(r1_region_te)

        r1_unicode = r1_region_te.find("ns:Unicode", namespaces=ns)
        self.assertIsNotNone(r1_unicode)
        self.assertIn(r1_unicode.text, (None, ""))

        # r2 line is updated.
        r2_line_te = r2_l1.find("ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(r2_line_te)
        self.assertEqual(
            r2_line_te.find("ns:Unicode", namespaces=ns).text,
            "Only region 2 should be updated",
        )

        # r2 region receives the collected region text.
        r2_region_te = r2.find("ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(r2_region_te)
        self.assertEqual(
            r2_region_te.find("ns:Unicode", namespaces=ns).text,
            "Only region 2 should be updated",
        )

    def test_insert_inferred_lines_works_without_namespace(self):
        xml_without_namespace = """
        <PcGts>
            <Page imageFilename="sample.png">
                <TextRegion id="r1">
                    <TextLine id="L1">
                        <Coords points="0,0 10,0 10,10 0,10"/>
                    </TextLine>
                </TextRegion>
            </Page>
        </PcGts>
        """

        xp = XMLProcessor.from_string(xml_without_namespace)

        updated_count = xp.insert_inferred_lines(
            xp.root,
            {("r1", "L1"): "No namespace text"},
        )

        self.assertEqual(updated_count, 1)

        root = ET.fromstring(xp.tree_to_string())

        line = root.find(".//TextLine[@id='L1']")
        line_te = line.find("TextEquiv")

        self.assertIsNotNone(line_te)
        self.assertEqual(line_te.find("Unicode").text, "No namespace text")

        region = root.find(".//TextRegion[@id='r1']")
        region_te = region.find("TextEquiv")

        self.assertIsNotNone(region_te)
        self.assertEqual(region_te.find("Unicode").text, "No namespace text")


if __name__ == "__main__":
    unittest.main()