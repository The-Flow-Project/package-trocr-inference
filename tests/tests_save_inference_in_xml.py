def test_write_to_xml_files(self):
    inferred_lines = {"TextRegion_1663284780281_22l2": "This is a test inference result."}
    self.inference.write_to_xml_files(inferred_lines, [self.test_xml_file])

    inferred_xml_file = os.path.join(self.test_directory,
                                     self.test_repo_base_path,
                                     self.test_out_path,
                                     "1155140_0001_47389007.xml")
    self.assertTrue(os.path.exists(inferred_xml_file))
    with open(inferred_xml_file, "r") as f:
        content = f.read()
    self.assertIn("This is a test inference result.", content)