(ns dl4clj.labels-source-test
  (:require [clojure.test :refer :all]
            [dl4clj.text.documentiterator.labels-source :refer :all])
  (:import [org.deeplearning4j.text.documentiterator LabelsSource]))

(def source-with-labels (labels-source ["label1" "label2"]))

(deftest label-source-test
  (testing "the label-source ns"
    (is (= org.deeplearning4j.text.documentiterator.LabelsSource (type (labels-source))))
    (is (= org.deeplearning4j.text.documentiterator.LabelsSource (type (labels-source ["label1" "label2"]))))
    (is (= ["label1" "label2"] (get-labels source-with-labels)))
    (is (= 2 (get-number-of-labels-used source-with-labels)))))
