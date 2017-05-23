(ns dl4clj.datasets-test
  (:require [clojure.test :refer :all]
            [dl4clj.datasets.datavec :refer :all]
            [dl4clj.datasets.rearrange :refer :all]
            [dl4clj.datasets.fetchers.default-dataset-fetchers :refer :all]
            [dl4clj.datasets.fetchers.base-data-fetcher :refer :all]))

(deftest dataset-fetchers-test
  (testing "dataset fetchers"
    (is (= org.deeplearning4j.datasets.fetchers.IrisDataFetcher (type (iris-fetcher))))
    (is (= org.deeplearning4j.datasets.fetchers.MnistDataFetcher (type (mnist-fetcher))))
    (is (= org.deeplearning4j.datasets.fetchers.MnistDataFetcher
           (type (mnist-fetcher :binarize? true))))
    (is (= org.deeplearning4j.datasets.fetchers.MnistDataFetcher
           (type
            (mnist-fetcher :binarize? true :train? true :shuffle? true :rng-seed 123))))
    (is (= java.lang.Integer (type (cursor (iris-fetcher)))))
    (is (= java.lang.Boolean (type (has-more? (iris-fetcher)))))
    (is (= java.lang.Integer (type (input-column-length (iris-fetcher)))))
    (is (= org.deeplearning4j.datasets.fetchers.IrisDataFetcher
           (type (reset-fetcher! (iris-fetcher)))))
    (is (= java.lang.Integer (type (n-examples-in-ds (iris-fetcher)))))
    (is (= java.lang.Integer (type (n-outcomes-in-ds (iris-fetcher)))))))

(deftest iterator-creation-test
  (testing "the creation of record readers dataset iterators"
    ;; lets test bottom level first then work up to this
    ))































(deftest rearrange-test
  (testing "the rearrange ns"
    ;; currently can't accurately test without the unstructed dataset these fns want

    #_(let [formatter (new-unstructured-formatter
                       :destination-root-dir (str "resources/tmp/rearrange/"
                                                  (java.util.UUID/randomUUID))
                       :src-root-dir "resources/poker"
                       :labeling-type :name
                       :percent-train 50)]
        (is (= org.deeplearning4j.datasets.rearrange.LocalUnstructuredDataFormatter
               (type formatter)))
        (is (= "" (get-new-destination :unstructured-formatter formatter
                                       :file-path (str "resources/tmp/rearrange/"
                                                       (java.util.UUID/randomUUID))
                                       :train? true)))

        )))
