#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use Test::More qw(no_plan);
BEGIN { use_ok('AI::NeuralNet::SOM::Rect') };

######
use Data::Dumper;

{
    my $nn = new AI::NeuralNet::SOM::Rect (output_dim => "5x6",
					   input_dim  => 3);
    ok ($nn->isa ('AI::NeuralNet::SOM::Rect'), 'class');
    is ($nn->{_X}, 5, 'X');
    is ($nn->{_Y}, 6, 'Y');
    is ($nn->{_Z}, 3, 'Z');
    is ($nn->radius, 2.5, 'radius');
}

{
    my $nn = new AI::NeuralNet::SOM::Rect (output_dim => "5x6",
					   input_dim  => 3);
    $nn->initialize;
#    print Dumper $nn;
#    exit;

    my @vs = ([ 3, 2, 4 ], [ -1, -1, -1 ], [ 0, 4, -3]);
    $nn->train (400, @vs);

    foreach my $v (@vs) {
	ok (_find ($v, $nn->map), 'found learned vector '. join (",", @$v));
    }

sub _find {
    my $v = shift;
    my $m = shift;

    use AI::NeuralNet::SOM::Utils;
    foreach my $x ( 0 .. 4 ) {
	foreach my $y ( 0 .. 5 ) {
	    return 1 if AI::NeuralNet::SOM::Utils::vector_distance ($m->[$x]->[$y], $v) < 0.01;
	}
    }
    return 0;
}


    ok ($nn->as_string, 'pretty print');
    ok ($nn->as_data, 'raw format');

#    print $nn->as_string;
}

__END__

