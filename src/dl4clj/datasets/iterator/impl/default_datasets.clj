(ns dl4clj.datasets.iterator.impl.default-datasets
  (:refer-clojure :exclude [reset!])
  (:import [org.deeplearning4j.datasets.iterator.impl
            CifarDataSetIterator IrisDataSetIterator LFWDataSetIterator
            MnistDataSetIterator RawMnistDataSetIterator]
           [java.util Random])
  (:require [dl4clj.utils :refer [contains-many?]]
            [datavec.api.split :refer :all]
            [nd4clj.linalg.api.ds-iter :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prebuilt dataset iterators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-cifar-data-set-iterator
  "Load the images from the cifar dataset,

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/CifarDataSetIterator.html
  and: https://github.com/szagoruyko/cifar.torch

  :batch-size (int), the batch size
  :n-examples (int), the number of examples from the ds to include in the iterator
  :img-dim (vector), desired dimensions of the images
   - should contain 3 ints
  :train? (boolean), are we training or testing?
  :use-special-pre-process-cifar? (boolean), are we going to use the predefined preprocessor built for this dataset
   - There is a special preProcessor used to normalize the dataset based on Sergey Zagoruyko example https://github.com/szagoruyko/cifar.torch
  :img-transform (map) config map for an image-transformation (as of writing this doc string, not implemented)
  :n-possible-labels (int), specify the number of possible outputs/tags/classes for a given image"
  [& {:keys [batch-size n-examples img-dims train?
             use-special-pre-process-cifar?
             n-possible-labels img-transform]
      :as opts}]
  (let [img (int-array img-dims)]
   (cond (contains-many? opts :batch-size :n-examples :img-dims :n-possible-labels
                        :img-transform :use-special-pre-process-cifar? :train?)
        (CifarDataSetIterator. batch-size n-examples img n-possible-labels
                               img-transform use-special-pre-process-cifar? train?)
        (contains-many? opts :batch-size :n-examples :img-dims :use-special-pre-process-cifar? :train?)
        (CifarDataSetIterator. batch-size n-examples img use-special-pre-process-cifar? train?)
        (contains-many? opts :batch-size :n-examples :img-dims :train?)
        (CifarDataSetIterator. batch-size n-examples img train?)
        (contains-many? opts :batch-size :n-examples :img-dims)
        (CifarDataSetIterator. batch-size n-examples img)
        (contains-many? opts :batch-size :n-examples :train?)
        (CifarDataSetIterator. batch-size n-examples train?)
        (contains-many? opts :batch-size :img-dims)
        (CifarDataSetIterator. batch-size img)
        (contains-many? opts :batch-size :n-examples)
        (CifarDataSetIterator. batch-size n-examples)
        :else
        (assert (and (contains? opts :batch-size)
                     (or (contains? opts :img-dims)
                         (contains? opts :n-examples)))
                "you must provide atleast a batch size and number of examples or a batch size and the desired demensions of the images"))))

(defn new-iris-data-set-iterator
  "IrisDataSetIterator handles traversing through the Iris Data Set.

  :batch (int), size of the batch
  :n-examples (int), number of examples to iterator over

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/IrisDataSetIterator.html"
  [& {:keys [batch n-examples]}]
  (IrisDataSetIterator. batch n-examples))

(defn new-lfw-data-set-iterator
  "Creates a dataset iterator for the LFW image dataset.

  :img-dims (int-array), desired dimensions of the images
  :batch-size (int), the batch size
  :n-examples (int), number of examples to take from the dataset
  :use-subset? (boolean), use a subset of the dataset or the whole thing
  :train? (boolean, are we training a net or testing it
  :split-train-test (double), the division between training and testing datasets
  :n-labels (int), the number of possible classifications for a single image
  :rng (anything), by supplying this key when calling this fn, it creates a new
  instance of java.util.Random
  :label-generator (label generator), call (new-parent-path-label-generator) or
   (new-pattern-path-label-generator opts) to create a label generator to use
  :image-transform (map), a transform to apply to the images,
   - as of writing this doc string, this functionality not implemented

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/LFWDataSetIterator.html"
  [& {:keys [img-dims batch-size n-examples use-subset? train? split-train-test
             n-labels rng label-generator image-transform]
      :as opts}]
  (let [img (int-array img-dims)]
   (cond (contains-many? opts :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :label-generator :train? :split-train-test :rng :image-transform)
        (LFWDataSetIterator. batch-size n-examples img n-labels use-subset?
                             label-generator train? split-train-test image-transform
                             (new Random))
        (contains-many? opts :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :label-generator :train? :split-train-test :rng)
        (LFWDataSetIterator. batch-size n-examples img n-labels use-subset?
                             label-generator train? split-train-test (new Random))
        (contains-many? opts :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :train? :split-train-test :rng)
        (LFWDataSetIterator. batch-size n-examples img n-labels use-subset?
                             train? split-train-test (new Random))
        (contains-many? opts :batch-size :n-examples :n-labels :train? :split-train-test)
        (LFWDataSetIterator. batch-size n-examples n-labels train? split-train-test)
        (contains-many? opts :batch-size :n-examples :img-dims :train? :split-train-test)
        (LFWDataSetIterator. batch-size n-examples img train? split-train-test)
        (contains-many? opts :batch-size :n-examples :img-dims)
        (LFWDataSetIterator. batch-size n-examples img)
        (contains-many? opts :batch-size :img-dims :use-subset?)
        (LFWDataSetIterator. batch-size img use-subset?)
        (contains-many? opts :batch-size :n-examples)
        (LFWDataSetIterator. batch-size n-examples)
        (contains? opts :img-dims)
        (LFWDataSetIterator. img)
        :else
        (assert false "you must supply atleast the desired image dimensions for the data"))))

(defn new-mnist-data-set-iterator
  "creates a dataset iterator for the Mnist dataset

  :batch-size (int), the batch size
  :train? (boolean), training or testing
  :seed (int), int used to randomize the dataset
  :n-examples (int), the overall number of examples
  :binarize? (boolean), whether to binarize mnist or not
  :shuffle? (boolean), whether to shuffle the dataset or not
  :rng-seed (long), random number generator seed to use when shuffling examples
  :batch (int), size of each patch

  - supplying batch-size will retrieve the entire dataset where as batch will get a subset

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/MnistDataSetIterator.html"
  [& {:keys [batch-size train? seed n-examples binarize? shuffle? rng-seed batch]
      :as opts}]
  (cond (contains-many? opts :batch-size :train? :seed)
        (MnistDataSetIterator. batch-size train? seed)
        (contains-many? opts :batch :n-examples :binarize? :train? :shuffle? :rng-seed)
        (MnistDataSetIterator. batch n-examples binarize? train? shuffle? rng-seed)
        (contains-many? opts :batch :n-examples :binarize?)
        (MnistDataSetIterator. batch n-examples binarize?)
        (contains-many? opts :batch :n-examples)
        (MnistDataSetIterator. batch n-examples)
        :else
        (assert false "you must atleast supply a batch and number of examples")))

(defn new-raw-mnist-data-set-iterator
  "Mnist data with scaled pixels

  :batch (int) size of each patch
  :n-examples (int), the overall number of examples

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/RawMnistDataSetIterator.html"
  [& {:keys [batch n-examples]}]
  (RawMnistDataSetIterator. batch n-examples))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; api fns not specified in nd4clj.linalg.api.ds-iter
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn has-next?
  [iter]
  (.hasNext iter))

(defn next-data-point
  [iter]
  (.next iter))

(defn test-cifar-iter!
  [& {:keys [iter n-examples batch-size]
      :as opts}]
  (cond (contains-many? opts :iter :n-examples :batch-size)
        (doto iter (.test n-examples batch-size))
        (contains-many? opts :iter :n-examples)
        (doto iter (.test n-examples))
        (contains? opts :iter)
        (doto iter (.test))
        :else
        (assert false "you must provide a cifar iterator")))

(defn train-cifar-iter!
  [iter]
  (doto iter (.train)))

(defn get-feature-matrix
  [data-from-iter]
  (.getFeatureMatrix data-from-iter))
