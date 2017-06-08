(ns dl4clj.datasets-test
  (:refer-clojure :exclude [reset!])
  (:require [clojure.test :refer :all]
            [dl4clj.datasets.datavec :refer :all]
            [dl4clj.datasets.rearrange :refer :all]
            [dl4clj.datasets.fetchers.default-dataset-fetchers :refer :all]
            [dl4clj.datasets.fetchers.base-data-fetcher :refer :all]
            [dl4clj.datasets.iterator.iterators :refer :all]
            [dl4clj.datasets.iterator.impl.default-datasets :refer :all]
            [dl4clj.datasets.iterator.impl.list-data-set-iterator :refer :all]
            [dl4clj.datasets.iterator.impl.move-window-data-set-fetcher :refer :all]
            [dl4clj.datasets.iterator.impl.multi-data-set-iterator-adapter :refer :all]
            [dl4clj.datasets.iterator.impl.singleton-multi-data-set-iterator :refer :all]
            [datavec.api.split :refer :all]
            [nd4clj.linalg.dataset.api.pre-processors :refer :all]
            [nd4clj.linalg.api.ds-iter :refer :all]
            [nd4clj.linalg.dataset.api.ds-preprocessor :refer :all]
            [datavec.api.writeable :refer :all]
            [datavec.api.records.readers :refer :all]
            [datavec.api.records.interface :refer :all]
            [dl4clj.utils :refer [array-of]])
  ;; image transforms have not been implemented so importing this default one for testing
  ;; https://deeplearning4j.org/datavecdoc/org/datavec/image/transform/package-summary.html
  (:import [org.datavec.image.transform ColorConversionTransform]))

(deftest dataset-fetchers-test
  (testing "dataset fetchers"
    ;; dl4clj.datasets.fetchers.default-dataset-fetchers
    (is (= org.deeplearning4j.datasets.fetchers.IrisDataFetcher (type (iris-fetcher))))
    (is (= org.deeplearning4j.datasets.fetchers.MnistDataFetcher (type (mnist-fetcher))))
    (is (= org.deeplearning4j.datasets.fetchers.MnistDataFetcher
           (type (mnist-fetcher :binarize? true))))
    (is (= org.deeplearning4j.datasets.fetchers.MnistDataFetcher
           (type
            (mnist-fetcher :binarize? true :train? true :shuffle? true :rng-seed 123))))
    ;; dl4clj.datasets.fetchers.base-data-fetcher
    (is (= java.lang.Integer (type (fetcher-cursor (iris-fetcher)))))
    (is (= java.lang.Boolean (type (has-more? (iris-fetcher)))))
    (is (= java.lang.Integer (type (input-column-length (iris-fetcher)))))
    (is (= org.deeplearning4j.datasets.fetchers.IrisDataFetcher
           (type (reset-fetcher! (iris-fetcher)))))
    (is (= java.lang.Integer (type (n-examples-in-ds (iris-fetcher)))))
    (is (= java.lang.Integer (type (n-outcomes-in-ds (iris-fetcher)))))))

(deftest ds-iteration-creation-test
  (testing "the creation of dataset iterators"
    ;; dl4clj.datasets.iterator.impl.default-datasets
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
           (type (new-raw-mnist-data-set-iterator :batch 5 :n-examples 100))))

    ;; dl4clj.datasets.iterator.impl.list-data-set-iterator
    (is (= org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
           (type (new-list-data-set-iterator
                  :data-set [(new-raw-mnist-data-set-iterator :batch 5 :n-examples 100)]))))
    (is (= org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
           (type (new-list-data-set-iterator
                  :data-set [(new-raw-mnist-data-set-iterator :batch 5 :n-examples 100)]
                  :batch 6))))

    ;; dl4clj.datasets.iterator.impl.move-window-data-set-fetcher
    ;; figure out what  Only rotating matrices means
    #_(is (= "" (type (new-moving-window-data-set-fetcher
                     :data-set (next-data-point (reset-fetcher! (new-mnist-data-set-iterator :batch 5 :n-examples 100)))
                     :window-rows 1
                     :window-columns 1))))
    #_(is (= "" (type (fetch! :ds-fetcher "" :n-examples 10))))

    ;; dl4clj.datasets.iterator.impl.multi-data-set-iterator-adapter
    (is (= org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter
           (type (new-multi-data-set-iterator-adapter
                  (new-mnist-data-set-iterator :batch 5 :n-examples 100)))))

    ;; dl4clj.datasets.iterator.impl.singleton-multi-data-set-iterator
    (is (= org.deeplearning4j.datasets.iterator.impl.SingletonMultiDataSetIterator
           (type (new-singleton-multi-data-set-iterator
                  (next-data-point
                   (reset-fetcher!
                    (new-multi-data-set-iterator-adapter
                     (new-mnist-data-set-iterator :batch 5 :n-examples 100))))))))))

(deftest ds-iteration-interaction-fns-test
  (testing "the api fns for ds iterators"
    (let [iter (new-mnist-data-set-iterator :batch 5 :n-examples 100)
          iter-w-labels (new-lfw-data-set-iterator :img-dims [1 1 1] :batch-size 2
                                                   :n-examples 100 :train? true
                                                   :split-train-test 0.50 :n-labels 5
                                                   :use-subset? true :rng 123
                                                   :label-generator (new-parent-path-label-generator)
                                                   :img-transform (ColorConversionTransform.))
          cifar-iter (new-cifar-data-set-iterator :batch-size 2 :n-examples 100)]
      ;; nd4clj.linalg.api.ds-iter
      (is (= java.lang.Boolean (type (async-supported? iter))))
      (is (= java.lang.Integer (type (get-batch-size iter))))
      (is (= java.lang.Integer (type (get-current-cursor iter))))
      (is (= java.util.ArrayList (type (get-labels iter-w-labels))))
      (is (= org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
             (type (set-pre-processor!
                    :iter iter
                    :pre-processor (new-min-max-normalization-ds-preprocessor :min-val 0 :max-val 1)))))
      (is (= org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
             (type (new-min-max-normalization-ds-preprocessor :min-val 0 :max-val 1))
             (type (get-pre-processor iter))))
      (is (= java.lang.Integer (type (get-input-columns iter))))
      (is (= org.nd4j.linalg.dataset.DataSet
             (type (next-n-examples :iter iter-w-labels :n 2))))
      (is (= java.lang.Integer (type (get-num-examples iter))))
      (is (= (type iter-w-labels) (type (reset-iter! iter-w-labels))))
      (is (= java.lang.Boolean (type (reset-supported? iter-w-labels))))
      (is (= java.lang.Integer (type (get-total-examples iter))))
      (is (= java.lang.Integer (type (get-total-outcomes iter))))

      ;; dl4clj.datasets.iterator.impl.default-datasets
      (is (= java.lang.Boolean (type (has-next? iter))))
      (is (= org.nd4j.linalg.dataset.DataSet (type (next-data-point iter-w-labels))))
      ;; this is going to fail when this is running in an enviro with gpus or spark I think
      ;; will need to see if that is the case in some way
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (get-feature-matrix (next-data-point (reset-iter! iter-w-labels))))))
      (is (= (type cifar-iter) (type (train-cifar-iter! cifar-iter))))
      (is (= (type cifar-iter) (type (test-cifar-iter! :iter cifar-iter))))
      (is (= (type cifar-iter) (type (test-cifar-iter! :iter cifar-iter
                                                       :n-examples 100))))
      (is (= (type cifar-iter) (type (test-cifar-iter! :iter cifar-iter
                                                       :n-examples 100
                                                       :batch-size 5)))))))

(deftest label-generators-test
  (testing "the creation of label generators and their functionality"
    (is (= org.datavec.api.io.labels.ParentPathLabelGenerator
           (type (new-parent-path-label-generator))))
    (is (= org.datavec.api.io.labels.PatternPathLabelGenerator
           (type (new-pattern-path-label-generator :pattern "."))))
    (is (= org.datavec.api.io.labels.PatternPathLabelGenerator
           (type (new-pattern-path-label-generator :pattern "." :pattern-position 0))))
    (is (= org.datavec.api.writable.Text
           (type
            (get-label-for-path :label-generator (new-parent-path-label-generator)
                                :path "resources/paravec/labeled/finance"))))))

(deftest path-filter-tests
  (testing "the creation of path filters and their functionality"
    (is (= org.datavec.api.io.filters.BalancedPathFilter
           (type
            (new-balanced-path-filter :rng (new java.util.Random)
                                      :extensions [".txt"]
                                      :label-generator (new-parent-path-label-generator)
                                      :max-paths 1
                                      :max-labels 3
                                      :min-paths-per-label 1
                                      :max-paths-per-label 1
                                      :labels []))))
    (is (= org.datavec.api.io.filters.RandomPathFilter
           (type
            (new-random-path-filter :rng (new java.util.Random)
                                    :extensions [".txt"]
                                    :max-paths 2))))
    (is (= (type
            (array-of :data [(java.net.URI/create "foo")]
                      :java-type java.net.URI))
           (type
            (filter-paths :path-filter (new-random-path-filter :rng (new java.util.Random)
                                                               :extensions [".txt"]
                                                               :max-paths 2)
                          :paths [(java.net.URI/create "foo")]))))))



(deftest file-split-testing
  (testing "base level io stuffs"
    ;; file split
    (is (= org.datavec.api.split.FileSplit
           (type (new-filesplit :root-dir "resources/poker"))))
    (is (= org.datavec.api.split.FileSplit
           (type (new-filesplit :root-dir "resources/poker"
                                :rng-seed (new java.util.Random)))))
    (is (= org.datavec.api.split.FileSplit
           (type (new-filesplit :root-dir "resources/poker"
                                :allow-format ".csv"))))
    (is (= org.datavec.api.split.FileSplit
           (type (new-filesplit :root-dir "resources/poker"
                                :allow-format ".csv"
                                :recursive? true))))
    (is (= org.datavec.api.split.FileSplit
           (type (new-filesplit :root-dir "resources/poker"
                                :allow-format ".csv"
                                :rng-seed (new java.util.Random)))))
    (is (= java.io.File (type (get-root-dir (new-filesplit :root-dir "resources/poker")))))

    ;; collection input split
    (is (= org.datavec.api.split.CollectionInputSplit
           (type (new-collection-input-split :coll [(new java.net.URI "foo")
                                                    (new java.net.URI "baz")]))))

    ;; input stream input split
    (with-open [data (clojure.java.io/input-stream "resources/poker/poker-hand-testing.csv")
                other-data (clojure.java.io/input-stream "resources/poker/poker-hand-training.csv")]
      (is (= org.datavec.api.split.InputStreamInputSplit
             (type (new-input-stream-input-split
                    :in-stream data
                    :file-path "resources/poker/poker-hand-testing.csv"))))
      (is (= org.datavec.api.split.InputStreamInputSplit
             (type (new-input-stream-input-split
                    :in-stream data))))

      (is (= java.io.BufferedInputStream
             (type (get-is (new-input-stream-input-split
                            :in-stream data)))))
      (= org.datavec.api.split.InputStreamInputSplit
         (type (set-is! :input-stream-input-split (new-input-stream-input-split
                                                   :in-stream data)
                        :is other-data))))

    ;; list string input split
    (is (= org.datavec.api.split.ListStringSplit
           (type (new-list-string-split :data (list "foo" "baz")))))
    (is (= (list "foo" "baz")
           (get-list-string-split-data
            (new-list-string-split :data (list "foo" "baz")))))

    ;; numbered file input split
    (is (= org.datavec.api.split.NumberedFileInputSplit
           (type
            (new-numbered-file-input-split :base-string "f_%d.txt"
                                           :inclusive-min-idx 0
                                           :inclusive-max-idx 10))))

    ;; string split
    (is (= org.datavec.api.split.StringSplit
           (type (new-string-split :data "foo baz bar"))))
    (is (= "foo baz bar" (get-list-string-split-data
                          (new-string-split :data "foo baz bar"))))

    ;; transform split
    (is (= org.datavec.api.split.TransformSplit
           (type
            (new-transform-split :base-input-split (new-collection-input-split
                                                    :coll [(new java.net.URI "foo")
                                                           (new java.net.URI "baz")])
                                 :to-be-replaced "foo"
                                 :replaced-with "oof"))))

    ;; sample
    (is (= (type (new-collection-input-split
                  :coll [(new java.net.URI "foo")
                         (new java.net.URI "baz")]))
           (type (first (sample :split (new-collection-input-split
                                        :coll [(new java.net.URI "foo")
                                               (new java.net.URI "baz")])
                                :weights [50 50]

                                :path-filter (new-random-path-filter
                                              :rng (new java.util.Random)
                                              :extensions [".txt"]
                                              :max-paths 2))))))
    (is (= (type (new-collection-input-split
                  :coll [(new java.net.URI "foo")
                         (new java.net.URI "baz")]))
           (type (first (sample :split (new-collection-input-split
                                        :coll [(new java.net.URI "foo")
                                               (new java.net.URI "baz")])
                                :weights [50 50])))))
    (is (= 2 (count (sample :split (new-collection-input-split
                                    :coll [(new java.net.URI "foo")
                                           (new java.net.URI "baz")])
                            :weights [50 50]))))))

(deftest input-split-interface-testing
  (testing "the interfaces used by input splits"
    ;; datavec.api.writeable
    (let [f-split (new-filesplit :root-dir "resources/poker")]

      (is (= java.lang.Long (type (length f-split))))
      (is (= java.net.URI (type (first (locations f-split)))))
      (is (= org.datavec.api.util.files.UriFromPathIterator
             (type (locations-iterator f-split))))
      (is (= org.nd4j.linalg.collection.CompactHeapStringList$CompactHeapStringListIterator
             (type (locations-path-iterator f-split))))
      (is (= (type f-split) (type (reset-input-split! f-split)))))))

(deftest record-readers-test
  (testing "the creation of record readers"
    ;; datavec.api.records.readers
    ;; csv-nlines-seq-rr
    (is (= org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader
           (type (new-csv-nlines-seq-record-reader))))
    (is (= org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader
           (type (new-csv-nlines-seq-record-reader :n-lines-per-seq 5))))
    (is (= org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader
           (type (new-csv-nlines-seq-record-reader :n-lines-per-seq 5
                                                   :delimiter "," :skip-num-lines 1))))

    ;; csv-rr
    (is (= org.datavec.api.records.reader.impl.csv.CSVRecordReader
           (type (new-csv-record-reader))))
    (is (= org.datavec.api.records.reader.impl.csv.CSVRecordReader
           (type (new-csv-record-reader :skip-lines 1))))
    (is (= org.datavec.api.records.reader.impl.csv.CSVRecordReader
           (type (new-csv-record-reader :skip-lines 1 :delimiter ","))))
    (is (= org.datavec.api.records.reader.impl.csv.CSVRecordReader
           (type (new-csv-record-reader :skip-lines 1 :delimiter ","
                                        :strip-quotes nil))))

    ;; csv-seq-rr
    (is (= org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
           (type (new-csv-seq-record-reader))))
    (is (= org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
           (type (new-csv-seq-record-reader :skip-num-lines 1))))
    (is (= org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
           (type (new-csv-seq-record-reader :skip-num-lines 1 :delimiter ","))))

    ;; file-rr
    (is (= org.datavec.api.records.reader.impl.FileRecordReader
           (type (new-file-record-reader))))

    ;; line-rr
    (is (= org.datavec.api.records.reader.impl.LineRecordReader
           (type (new-line-record-reader))))

    ;; list-string-rr
    (is (= org.datavec.api.records.reader.impl.collection.ListStringRecordReader
           (type (new-list-string-record-reader))))))

(deftest record-readers-interface
  (testing "the api fns for record readers"
    (let [rr (new-file-record-reader)
          init-rr (initialize-rr! :rr rr :input-split
                                  (new-filesplit
                                   :root-dir "resources/poker/poker-hand-testing.csv"))]
      ;; these tests do not cover the entire ns but the most imporant fns
      (is (= java.lang.Boolean (type (has-next-record? init-rr))))
      (is (= org.datavec.api.records.impl.Record (type (next-record-with-meta! init-rr))))
      (is (= java.util.ArrayList (type (next-record! (reset-rr! init-rr))))))))

(deftest pre-processors-test
  (testing "testing the creation of pre-processors"
    ;; nd4clj.linalg.dataset.api.pre-processors
    (is (= org.nd4j.linalg.dataset.api.preprocessor.ImageFlatteningDataSetPreProcessor
           (type (new-image-flattening-ds-preprocessor))))
    (is (= org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
           (type (new-image-scaling-ds-preprocessor))))
    (is (= org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
           (type (new-image-scaling-ds-preprocessor :min-range 0 :max-range 150))))
    (is (= org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
           (type (new-vgg16-image-preprocessor))))
    (is (= org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
           (type (new-min-max-normalization-ds-preprocessor :min-val 0 :max-val 1))))
    (is (= org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
           (type (new-min-max-normalization-ds-preprocessor :min-val 7 :max-val 15))))
    (is (= org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
           (type (new-standardize-normalization-ds-preprocessor))))))


(deftest ds-iterators-test
  (testing "the creation of various dataset iterators"
    ;; dl4clj.datasets.iterator.iterators
    (let [iter (new-mnist-data-set-iterator :batch 5 :n-examples 100)
          pp1 (fit-iter! :normalizer (new-image-scaling-ds-preprocessor)
                         :ds-iter iter)
          pp2 (fit-iter! :normalizer (new-min-max-normalization-ds-preprocessor :min-val 0 :max-val 1)
                         :ds-iter iter)
          multi-iter (new-multi-data-set-iterator-adapter
                      (new-mnist-data-set-iterator :batch 5 :n-examples 100))]
      (is (= org.deeplearning4j.datasets.iterator.CombinedPreProcessor
             (type (new-combined-pre-processor :pre-processors {0 pp1
                                                                1 pp2}))))
      (is (= org.deeplearning4j.datasets.iterator.AsyncDataSetIterator
             (type (new-async-dataset-iterator :dataset-iterator iter))))
      (is (= org.deeplearning4j.datasets.iterator.AsyncDataSetIterator
             (type (new-async-dataset-iterator :dataset-iterator iter
                                                :que-size 10))))
      (is (= org.deeplearning4j.datasets.iterator.ExistingDataSetIterator
             (type (new-existing-dataset-iterator :dataset-iterator iter))))
      (is (= org.deeplearning4j.datasets.iterator.SamplingDataSetIterator
             (type
              (new-sampling-dataset-iterator :sampling-source iris-ds
                                             :batch-size 10
                                             :total-n-samples 10))))
      (is (= org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator
             (type (new-reconstruction-dataset-iterator :dataset-iterator iter))))
      ;; again rotating matrices error
      #_(is (= "" (type (new-moving-window-base-dataset-iterator :batch-size 10
                                                               :n-examples 10
                                                               :dataset iris-ds
                                                               :window-rows 2
                                                               :window-columns 2))))
      (is (= org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator
             (type (new-async-multi-dataset-iterator
                    :multi-dataset-iterator multi-iter
                    :que-length 10))))
      (is (= org.deeplearning4j.datasets.iterator.IteratorDataSetIterator
             (type (new-iterator-dataset-iterator :dataset iter :batch-size 10))))
      (is (= org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
             (type (new-multiple-epochs-iterator :dataset-iterator iter :n-epochs 1)))))))

(deftest rr-ds-iterator-test
  (testing "the creation of record reader dataset iterators"
    ;; dl4clj.datasets.datavec
    (let [rr (new-csv-record-reader)
          seq-rr (new-csv-seq-record-reader)]
      (is (= org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
             (type
              (new-record-reader-dataset-iterator :record-reader rr
                                                  :batch-size 10))))
      (is (= org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
             (type
              (new-record-reader-dataset-iterator :record-reader rr
                                                  :batch-size 10
                                                  :label-idx 6
                                                  :n-possible-labels 10))))
      (is (= org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
             (type
              (new-record-reader-dataset-iterator :record-reader rr
                                                  :batch-size 10
                                                  :label-idx-from 0
                                                  :label-idx-to 7
                                                  :regression? true))))
      (is (= org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
             (type
              (new-record-reader-dataset-iterator :record-reader rr
                                                  :batch-size 10
                                                  :label-idx 6
                                                  :n-possible-labels 10
                                                  :max-num-batches 2))))

      (is (= org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
           (type (new-seq-record-reader-dataset-iterator
                  :record-reader seq-rr
                  :mini-batch-size 5
                  :n-possible-labels 10
                  :label-idx 10
                  :regression? false))))
      (is (= org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
             (type (new-seq-record-reader-dataset-iterator
                    :record-reader seq-rr
                    :mini-batch-size 5
                    :n-possible-labels 10
                    :label-idx 10))))
      (is (= org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
             (type (new-record-reader-multi-dataset-iterator
                    :alignment-mode :equal-length
                    :batch-size 10
                    :add-seq-reader {:reader-name "foo" :record-reader seq-rr}
                    :add-reader {:reader-name "baz" :record-reader rr}
                    :add-input {:reader-name "baz" :first-column 0 :last-column 11})))))))

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
