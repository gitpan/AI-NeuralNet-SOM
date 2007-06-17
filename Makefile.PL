use 5.008008;
use ExtUtils::MakeMaker;

WriteMakefile(
    NAME              => 'AI::NeuralNet::SOM',
    VERSION_FROM      => 'lib/AI/NeuralNet/SOM.pm',
    PREREQ_PM         => {},
    NORECURS => 1,
    ($] >= 5.005 ?     ## Add these new keywords supported since 5.005
      (ABSTRACT_FROM  => 'lib/AI/NeuralNet/SOM.pm', # retrieve abstract from module
       AUTHOR         => 'Robert Barta <rho@devc.at>') : ()),
);


package MY;

sub depend {
    return <<MAKE

debian: 
	(cd ; cd packages ; rm -rf libai-neuralnet-som-perl* AI-NeuralNet-SOM* ; dh-make-perl --build -cpan AI::NeuralNet::SOM::Hexa  )

upload:
	cpan-upload-http AI-NeuralNet-SOM-\${VERSION}.tar.gz
MAKE
}
