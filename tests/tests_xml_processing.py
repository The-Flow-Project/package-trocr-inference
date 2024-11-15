import unittest
import os
import xml.etree.ElementTree as ET

from flow_inference.xml_processing import XMLProcessor


class TestXMLProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a sample XML file to use in tests."""
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        cls.test_xml_without_inference_escriptorium = os.path.join(current_dir, '..', 'test_data', 'xml', '1_0054.xml')
        cls.test_xml_with_inference_escriptorium = os.path.join(current_dir,
                                                                '..',
                                                                'test_data',
                                                                'xml_with_inference',
                                                                '1_0054.xml')
        cls.test_xml_without_inference_transkribus = os.path.join(current_dir,
                                                                  '..',
                                                                  'test_data',
                                                                  'xml',
                                                                  '1155140_0001_47389007.xml')
        cls.test_xml_with_inference_transkribus = os.path.join(current_dir,
                                                               '..',
                                                               'test_data',
                                                               'xml_with_inference',
                                                               '1155140_0001_47389007.xml')

        cls.output_path = os.path.join("/tmp", "test_xml_with_inference.xml")

    def test_parse_xml(self):
        """Test parsing the XML file."""
        tree = XMLProcessor.parse_xml(self.test_xml_without_inference_escriptorium)
        root = tree.getroot()
        self.assertIsNotNone(root)

    def test_find_line_id(self):
        """Test extracting the 'id' attribute from a TextLine element."""
        tree = XMLProcessor.parse_xml(self.test_xml_without_inference_escriptorium)
        root = tree.getroot()
        namespace = root.tag.split("}")[0].strip("{")
        ns = {'ns': namespace}

        # Use the prefix in the find method
        text_line = root.find(".//ns:TextLine[@id='line_1452904354230_8']", namespaces=ns)

        # Ensure text_line was found and proceed to check the id
        if text_line is not None:
            line_id = XMLProcessor.find_line_id(text_line)
            self.assertEqual(line_id, "line_1452904354230_8")

    def test_create_text_equiv_element(self):
        """Test creating the <TextEquiv> element."""
        text = "Sample text"

        # Initialize XMLProcessor and get the namespace
        processor = XMLProcessor(self.test_xml_without_inference_escriptorium)
        tree = XMLProcessor.parse_xml(self.test_xml_without_inference_escriptorium)
        root = tree.getroot()
        namespace = root.tag.split("}")[0].strip("{")
        ns = {'ns': namespace}

        # Create the <TextEquiv> element
        text_equiv = processor.create_text_equiv_element(text)

        # Check the <TextEquiv> tag (with the namespace)
        self.assertEqual(text_equiv.tag, f"{{{ns['ns']}}}TextEquiv")

        # Check the <Unicode> child element (with the namespace)
        unicode_element = text_equiv.find(f"{{{ns['ns']}}}Unicode")
        self.assertIsNotNone(unicode_element)
        self.assertEqual(unicode_element.text, text)

    def test_insert_text_equiv_tags(self):
        """Test inserting <TextEquiv> tags under the correct <TextLine> elements."""
        inferred_lines = {"line_1452904354230_8": "den Schoppfen des gerichtes der stat zu Wyen",
                          "line_1452904384999_9": "L fre vns hat fürgelegt heinr Trospg vns burger , wie daz "}

        processor = XMLProcessor(self.test_xml_without_inference_escriptorium)
        tree = XMLProcessor.parse_xml(self.test_xml_without_inference_escriptorium)
        root = tree.getroot()
        namespace = root.tag.split("}")[0].strip("{")
        ns = {'ns': namespace}
        processor.insert_text_equiv_tags(root, inferred_lines)

        # Save the updated tree to a temporary file
        temp_file = "/tmp/test_output.xml"
        tree.write(temp_file, encoding="utf-8", xml_declaration=True)

        # Reparse the saved file to verify changes
        updated_tree = XMLProcessor.parse_xml(temp_file)
        updated_root = updated_tree.getroot()

        # Check that the <TextEquiv> elements were added
        text_line_1 = updated_root.find(".//ns:TextLine[@id='line_1452904354230_8']", namespaces=ns)
        text_equiv_1 = text_line_1.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(text_equiv_1)
        self.assertEqual(text_equiv_1.find(".//ns:Unicode", namespaces=ns).text,
                         "den Schoppfen des gerichtes der stat zu Wyen")

        text_line_2 = updated_root.find(".//ns:TextLine[@id='line_1452904457580_15']", namespaces=ns)
        text_equiv_2 = text_line_2.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNone(text_equiv_2)

        text_line_3 = updated_root.find(".//ns:TextLine[@id='line_1452904384999_9']", namespaces=ns)
        text_equiv_3 = text_line_3.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(text_equiv_3)
        self.assertEqual(text_equiv_3.find(".//ns:Unicode", namespaces=ns).text,
                         "L fre vns hat fürgelegt heinr Trospg vns burger , wie daz ")

    def test_save_xml(self):
        """Test saving the XML tree to a file."""
        tree = XMLProcessor.parse_xml(self.test_xml_without_inference_escriptorium)
        root = tree.getroot()
        namespace = root.tag.split("}")[0].strip("{")
        ns = {'ns': namespace}

        inferred_lines = {"line_1452904354230_8": "den Schoppfen des gerichtes der stat zu Wyen",
                          "line_1452904384999_9": "L fre vns hat fürgelegt heinr Trospg vns burger , wie daz "}
        processor = XMLProcessor(self.test_xml_without_inference_escriptorium)
        processor.insert_text_equiv_tags(root, inferred_lines)

        # Save to a new file
        processor.save_xml(tree, self.output_path)

        # Verify that the output file was created and contains the changes
        self.assertTrue(os.path.exists(self.output_path))

        # Check if the saved XML contains the expected text
        tree = ET.parse(self.output_path)
        updated_root = tree.getroot()
        text_line_1 = updated_root.find(".//ns:TextLine[@id='line_1452904354230_8']", namespaces=ns)
        text_equiv_1 = text_line_1.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(text_equiv_1)
        self.assertEqual(text_equiv_1.find(".//ns:Unicode", namespaces=ns).text,
                         "den Schoppfen des gerichtes der stat zu Wyen")

        text_line_2 = updated_root.find(".//ns:TextLine[@id='line_1452904457580_15']", namespaces=ns)
        text_equiv_2 = text_line_2.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNone(text_equiv_2)

        text_line_3 = updated_root.find(".//ns:TextLine[@id='line_1452904384999_9']", namespaces=ns)
        text_equiv_3 = text_line_3.find(".//ns:TextEquiv", namespaces=ns)
        self.assertIsNotNone(text_equiv_3)
        self.assertEqual(text_equiv_3.find(".//ns:Unicode", namespaces=ns).text,
                         "L fre vns hat fürgelegt heinr Trospg vns burger , wie daz ")


if __name__ == "__main__":
    unittest.main()
