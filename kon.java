public class kon{
    public static void main(String[] args) {
        int arr[]={1,2,7,34,578,12,45};
        quicksort(arr,0,arr.length-1);
        for(int x:arr){
            System.out.print(x+" ");
        }
}
public static int partition(int arr[],int low,int high){
    int p=low;
    int i=low;
    int j=high;
    while(i<j){
        while(arr[i]<=arr[p]&&i<high){
            i++;
        }
        while(arr[j]>arr[p] && j>low){
            j--;
        }
        if(i<j) swap(arr,i,j);
    }
    swap(arr,j,p);
    return j;
}
public static void quicksort(int arr[],int low,int high){
    if(low<high){
        int j=partition(arr,low,high);
        quicksort(arr,low,j-1);
        quicksort(arr, j+1,high);
    }
    return;
}
public static void swap(int arr[],int a,int b){
    int temp;
    temp=arr[a];
    arr[a]=arr[b];
    arr[b]=temp;
}
}
