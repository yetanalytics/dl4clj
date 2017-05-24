(ns dl4clj.datasets-test
  (:require [clojure.test :refer :all]
            [dl4clj.datasets.datavec :refer :all]
            [dl4clj.datasets.rearrange :refer :all]
            [dl4clj.datasets.fetchers.default-dataset-fetchers :refer :all]
            [dl4clj.datasets.fetchers.base-data-fetcher :refer :all]
            [dl4clj.datasets.iterator.iterators :refer :all]
            [dl4clj.datasets.iterator.impl.default-datasets :refer :all]
            [datavec.api.split :refer :all]
            [nd4clj.linalg.api.ds-iter :refer :all])
  ;; image transforms have not been implemented so importing this default one for testing
  ;; https://deeplearning4j.org/datavecdoc/org/datavec/image/transform/package-summary.html
  (:import [org.datavec.image.transform ColorConversionTransform]))

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

(deftest ds-iteration-creation-test
  (testing "the creation of dataset iterators"
    ;; cifar dataset
    (is (= org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
           (type (new-cifar-data-set-iterator :batch-size 2 :n-examples 100))))
    (is (= org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
           (type (new-cifar-data-set-iterator :batch-size 2 :img-dims [1 1 1]))))
    (is (= org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
           (type (new-cifar-data-set-iterator :batch-size 2 :n-examples 100 :train? true))))
    (is (= org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
           (type (new-cifar-data-set-iterator :batch-size 2 :n-examples 100
                                              :train? true :img-dims [1 1 1]))))
    (is (= org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
           (type (new-cifar-data-set-iterator :batch-size 3 :n-examples 100
                                              :train? true :img-dims [3 3 3]
                                              :use-special-pre-process-cifar? true))))
    (is (= org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
           (type (new-cifar-data-set-iterator :batch-size 3 :n-examples 100
                                              :train? true :img-dims [3 3 3]
                                              :use-special-pre-process-cifar? true
                                              :n-possible-labels 5
                                              :img-transform (ColorConversionTransform.)))))

    ;; iris dataset
    (is (= org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
           (type (new-iris-data-set-iterator :batch 2 :n-examples 100))))

    ;; lfwd
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :img-dims [1 1 1]))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :batch-size 2 :n-examples 100))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                            :use-subset? true))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                            :n-examples 100))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                            :n-examples 100 :train? true
                                            :split-train-test 0.50))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :n-labels 5 :batch-size 2
                                            :n-examples 100 :train? true
                                            :split-train-test 0.50))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                            :n-examples 100 :train? true
                                            :split-train-test 0.50 :n-labels 5
                                            :use-subset? true :rng 123))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                            :n-examples 100 :train? true
                                            :split-train-test 0.50 :n-labels 5
                                            :use-subset? true :rng 123))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                            :n-examples 100 :train? true
                                            :split-train-test 0.50 :n-labels 5
                                            :use-subset? true :rng 123
                                            :label-generator (new-parent-path-label-generator)))))
    (is (= org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
           (type (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                            :n-examples 100 :train? true
                                            :split-train-test 0.50 :n-labels 5
                                            :use-subset? true :rng 123
                                            :label-generator (new-parent-path-label-generator)
                                            :img-transform (ColorConversionTransform.)))))

    ;;mnist
    (is (= org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
           (type (new-mnist-data-set-iterator :batch-size 5 :train? true
                                              :seed 123))))
    (is (= org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
           (type (new-mnist-data-set-iterator :batch 5 :n-examples 100))))
    (is (= org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
           (type (new-mnist-data-set-iterator :batch 5 :n-examples 100 :binarize? true))))
    (is (= org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
           (type (new-mnist-data-set-iterator :batch 5 :n-examples 100 :binarize? true
                                               :train? true :shuffle? true :rng-seed 123))))

    ;; raw mnist
    (is (= org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator
           (type (new-raw-mnist-data-set-iterator :batch 5 :n-examples 100))))))

(deftest ds-iteration-interaction-fns-test
  (testing "the api fns for ds iterators"
    (let [iter (new-mnist-data-set-iterator :batch 5 :n-examples 100)
          iter-w-labels (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                                   :n-examples 100 :train? true
                                                   :split-train-test 0.50 :n-labels 5
                                                   :use-subset? true :rng 123
                                                   :label-generator (new-parent-path-label-generator)
                                                   :img-transform (ColorConversionTransform.))
          iter-w-pp (new-cifar-data-set-iterator :batch-size 3 :n-examples 100
                                                 :train? true :img-dims [3 3 3]
                                                 :use-special-pre-process-cifar? true)]
      (is (= java.lang.Boolean (type (async-supported? iter))))
      (is (= java.lang.Integer (type (get-batch-size iter))))
      (is (= java.lang.Integer (type (get-current-cursor iter))))
      (is (= java.util.ArrayList (type (get-labels iter-w-labels))))

      (is (= "" (type (get-pre-processor iter-w-pp))))
      (is (= java.lang.Boolean (type (has-next? iter))))
      )

    ))

(deftest rr-ds-iterator-creation-test
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
