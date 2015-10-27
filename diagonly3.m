function v = diagonly3(A,B,C)
sizeA = size(A);
sizeB = size(B);
sizeC = size(C);

v1 = zeros(sizeB(1),1);
v = zeros(sizeA(1),1);

if sizeA(2) ~= sizeB(1)
    error('Matricies different sizes')
elseif sizeB(2) ~= sizeC(1)
    error('Matricies different sizes')
else
    
    for i = 1:sizeA(1)        
        a = 0;
        
        for j = 1:sizeA(2)
            for k = 1:sizeB(2)
                a = a + A(i,j)*B(j,k)*C(k,i);
            end
        end
        v(i) = a;
    end
end
