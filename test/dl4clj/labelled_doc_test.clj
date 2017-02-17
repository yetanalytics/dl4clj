(ns dl4clj.labelled-doc-test
  (:require [clojure.test :refer :all]
            [dl4clj.text.documentiterator.labelled-document :refer :all])
  (:import [org.deeplearning4j.text.documentiterator LabelledDocument]))

(def empty-doc (labelled-document))
(def non-empty-doc (labelled-document "this is content" "this is a label"))

(deftest label-test
  (testing "the fns dealing with docs and their labels"
    (is (= "LabelledDocument(content=null, labels=[])" (str (labelled-document))))
    (is (= "LabelledDocument(content=null, labels=[])"  (str empty-doc)))
    (is (= org.deeplearning4j.text.documentiterator.LabelledDocument (type (labelled-document))))
    (is (= "LabelledDocument(content=this is content, labels=[this is a label])"
       (str (labelled-document "this is content" "this is a label"))))

    (is (= "this is content" (get-content non-empty-doc)))
    (is (= "this is a label" (get-label non-empty-doc)))))
